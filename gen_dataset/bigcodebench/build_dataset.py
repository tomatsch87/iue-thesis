"""
Generate BigCodeBench model outputs with prompt metadata and placeholder labels.

Default behavior:
- Loads BigCodeBench `subset=full`, `split=instruct`, `version=v0.1.4`.
- Produces `n=1` generation per task with defaults:
	`temperature=0.2`, `top_p=1.0`, `top_k=0`, `repetition_penalty=1.0`.
- vLLM defaults: `max_tokens=32000`, `max_model_len=32768`,
	`gpu_memory_utilization=0.85`, `tensor_parallel_size=4`, `dtype=bfloat16`.
- Applies a system instruction prefix by default (disable via `--disable-prefix`).
- Initializes `is_correct` as `False` placeholders; scoring is done by later scripts.
- If `--output-path` is omitted, writes to
	`/data/bigcodebench/bigcodebench_<split>_<model_stub>.jsonl`.

Important options:
- Dataset scope: `--subset`, `--version`, `--split`, `--start`, `--limit`.
- Prompting: `--instruction-prefix`, `--disable-prefix`, `--harmony-format`.
- Runtime/model: `--model`, `--tokenizer`, `--trust-remote-code`,
	`--enable-prefix-caching`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import textwrap
import time
from pathlib import Path
from typing import Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

from .vllm_runner import VLLMRunner

LOGGER = logging.getLogger(__name__)

HARMONY_ASSISTANT_MSG_RE = re.compile(
	r"<\|start\|>assistant<\|channel\|>(analysis|final)<\|message\|>(.*?)<\|end\|>",
	re.DOTALL,
)
HARMONY_CHANNEL_MARKER_RE = re.compile(r"<\|channel\|>(analysis|final)<\|message\|>", re.DOTALL)

INSTRUCT_DEFAULT_PREFIX = (
    "Please provide a self-contained Python script that solves the following problem in a fenced code block. "
	"The last code block in your response that is fenced using backticks should be your final answer:"
)
FUSION_DEFAULT_PREFIX = (
	"You are an expert Python programmer. You will be given a question (problem specification) "
	"and will generate a correct Python program that matches the specification and passes all tests."
)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate BigCodeBench completions with vLLM")
	parser.add_argument("--model", required=True, help="Model name or path usable by vllm")
	parser.add_argument("--tokenizer", default=None, help="Optional tokenizer name; defaults to model")
	parser.add_argument(
		"--output-path",
		type=Path,
		help="Full JSONL output path. If omitted, a file is created under --output-dir.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("/data/bigcodebench"),
		help="Directory for JSONL output.",
	)
	parser.add_argument("--subset", default="full", help="Dataset subset name (default: full)")
	parser.add_argument("--split", default="instruct", choices=["instruct", "complete"], help="Prompt split to use")
	parser.add_argument("--n", type=int, default=1, help="Number of generations per prompt")
	parser.add_argument("--temperature", type=float, default=0.2)
	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=0)
	parser.add_argument("--repetition-penalty", type=float, default=1.0)
	parser.add_argument("--max-tokens", type=int, default=32000)
	parser.add_argument("--max-model-len", type=int, default=32768)
	parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
	parser.add_argument("--tensor-parallel-size", type=int, default=4)
	parser.add_argument("--dtype", default="bfloat16")
	parser.add_argument("--enable-prefix-caching", action="store_true")
	parser.add_argument("--trust-remote-code", action="store_true")
	parser.add_argument("--start", type=int, default=None, help="Optional task index to start from (inclusive)")
	parser.add_argument("--limit", type=int, default=None, help="Optional limit of tasks (exclusive)")
	parser.add_argument(
		"--harmony-format",
		action="store_true",
		help="Treat outputs as OpenAI Harmony formatted responses for stripping.",
	)
	parser.add_argument(
		"--disable-prefix",
		action="store_true",
		help="Disable applying instruction prefix (raw prompts only)",
	)
	parser.add_argument(
		"--instruction-prefix",
		default=FUSION_DEFAULT_PREFIX,
		help="Instruction prefix applied when prefixes are enabled",
	)
	parser.add_argument(
		"--version",
		default="v0.1.4",
		help="BigCodeBench dataset version tag (default: v0.1.4)",
	)
	return parser.parse_args()


def _extract_harmony_output_stripped(text: str) -> str:
	parts: list[str] = []
	for m in HARMONY_ASSISTANT_MSG_RE.finditer(text):
		parts.append(m.group(2))
	clean = text
	clean = clean.replace("<|return|>", "")
	clean = re.sub(r"<\|start\|>assistant\s*", "", clean)
	clean = clean.replace("<|end|>", "")
	if not parts:
		idx = 0
		while True:
			m = HARMONY_CHANNEL_MARKER_RE.search(clean, idx)
			if not m:
				break
			body_start = m.end()
			next_marker = HARMONY_CHANNEL_MARKER_RE.search(clean, body_start)
			next_return = clean.find("<|return|>", body_start)
			candidates = [pos for pos in [next_marker.start() if next_marker else -1, next_return] if pos != -1]
			body_end = min(candidates) if candidates else len(clean)
			parts.append(clean[body_start:body_end])
			idx = body_end
	if parts:
		return "\n".join(p.strip() for p in parts if p)
	clean = re.sub(r"<\|start\|>assistant(?:<\|channel\|>[A-Za-z0-9_-]+)?<\|message\|>", "", clean)
	clean = re.sub(r"<\|channel\|>(analysis|final)<\|message\|>", "", clean)
	return clean.strip()


def extract_code_span(text: str) -> Optional[Tuple[int, int]]:
	"""Return (char_start, char_end) of the last fenced code block body."""
	fences = list(re.finditer(r"```", text))
	if len(fences) < 2:
		return None
	start_fence = fences[-2].start()
	end_fence = fences[-1].start()
	if start_fence >= end_fence:
		return None
	body_start = start_fence + 3
	newline_idx = text.find("\n", body_start, end_fence)
	if newline_idx != -1:
		body_start = newline_idx + 1
	body_end = end_fence
	if body_start >= body_end:
		return None
	return (body_start, body_end)


def compute_token_span(tokenizer, text: str, span: Tuple[int, int]) -> Tuple[int, int]:
	start_char, end_char = span
	encoding = tokenizer(
		text,
		return_offsets_mapping=True,
		add_special_tokens=False,
		padding=False,
		truncation=False,
	)
	offsets = encoding.get("offset_mapping")
	if offsets is None:
		return (0, 0)
	first = None
	last = None
	for idx, (start, end) in enumerate(offsets):
		if end <= start_char:
			continue
		if start >= end_char:
			break
		if first is None:
			first = idx
		last = idx
	if first is None or last is None:
		return (0, 0)
	return (int(first), int(last))


def _build_user_prompt(intruct: str, complete: str) -> str:
	docstring_match = re.search(r'"""(.*?)"""', complete, re.DOTALL)
	docstring_text = textwrap.dedent(docstring_match.group(1)).strip() if docstring_match else complete.strip()
	starter_match = re.search(r"(You should write self-contained code starting with:.*)$", intruct, re.DOTALL)
	starter_text = starter_match.group(1).strip() if starter_match else ""
	question = docstring_text
	if starter_text:
		question = f"{question}\n\n{starter_text}"

	prompt_lines = [f"### Question:\nWrite a self-contained Python function that solves the following problem:\n\n{question}\n"]
	prompt_lines.append("### Format: Enclose your code within delimiters as follows.\n")
	prompt_lines.append("```python\n# YOUR CODE HERE\n```\n")
	prompt_lines.append("### Answer: (use the provided format with backticks)\n\n")
	return "".join(prompt_lines)


def _build_chat_prompt(tokenizer, prompt: str, instruction_prefix: str, disable_prefix: bool) -> str:
	user_content = prompt.strip()
	messages = []
	if not disable_prefix and instruction_prefix:
		messages.append({"role": "system", "content": instruction_prefix})
	messages.append({"role": "user", "content": user_content})
	if tokenizer and getattr(tokenizer, "apply_chat_template", None):
		try:
			return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		except Exception:
			pass
	if not disable_prefix and instruction_prefix:
		return instruction_prefix + "\n\n" + user_content
	return user_content


def _resolve_output_path(args: argparse.Namespace) -> Path:
	if args.output_path:
		return args.output_path
	args.output_dir.mkdir(parents=True, exist_ok=True)
	model_stub = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
	extra = f"-{args.subset}" if args.subset != "full" else ""
	return args.output_dir / f"bigcodebench{extra}_{args.split}_{model_stub}.jsonl"


def load_bigcodebench(version: str, subset: str) -> list[dict]:
	extra = f"-{subset}" if subset != "full" else ""
	dataset_name = f"bigcode/bigcodebench{extra}"
	ds = load_dataset(dataset_name, split=version)
	return list(ds)


def main() -> None:
	args = parse_args()
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

	tokenizer_name = args.tokenizer or args.model
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=args.trust_remote_code)
	problems = load_bigcodebench(version=args.version, subset=args.subset)

	if args.start:
		problems = problems[args.start :]
	if args.limit:
		problems = problems[: args.limit]

	runner = VLLMRunner(
		model_name=args.model,
		tokenizer_name=tokenizer_name,
		n=args.n,
		temperature=args.temperature,
		top_p=args.top_p,
		top_k=args.top_k,
		max_tokens=args.max_tokens,
		max_model_len=args.max_model_len,
		gpu_memory_utilization=args.gpu_memory_utilization,
		tensor_parallel_size=args.tensor_parallel_size,
		dtype=args.dtype,
		repetition_penalty=args.repetition_penalty,
		enable_prefix_caching=args.enable_prefix_caching,
		trust_remote_code=args.trust_remote_code,
	)

	output_path = _resolve_output_path(args)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	try:
		with output_path.open("w", encoding="utf-8") as f:
			for idx, problem in enumerate(problems):
				problem_start = time.perf_counter()
				# On normal run use prompt_raw = problem["instruct_prompt"] or problem["complete_prompt"]
				prompt_raw = _build_user_prompt(problem["instruct_prompt"], problem["complete_prompt"]) if args.split == "instruct" else problem["complete_prompt"]
				prompt_text = _build_chat_prompt(tokenizer, prompt_raw, args.instruction_prefix, args.disable_prefix)
				prompt_enc = tokenizer(prompt_text, return_tensors="pt")
				prompt_token_ids = prompt_enc["input_ids"][0].tolist()
				prompt_token_count = len(prompt_token_ids)

				gens = runner.generate([prompt_text])
				if not gens:
					duration = time.perf_counter() - problem_start
					LOGGER.info("Skipped task %s in %.2fs (no generations)", problem["task_id"], duration)
					continue
				candidates = gens[0]
				output_texts: list[str] = []
				stripped_outputs: list[str] = []
				gen_token_ids: list[list[int]] = []
				for cand in candidates:
					cand_tokens = cand.get("token_ids") or []
					gen_token_ids.append(cand_tokens)
					decoded = tokenizer.decode(cand_tokens, skip_special_tokens=False) if cand_tokens else cand.get("text", "")
					output_texts.append(decoded)
					if args.harmony_format:
						stripped_outputs.append(_extract_harmony_output_stripped(decoded))
					else:
						stripped_outputs.append(cand.get("text", ""))

				code_spans = []
				code_texts: list[str] = []
				full_token_ids: list[list[int]] = []
				for out, cand_tokens in zip(output_texts, gen_token_ids):
					span = extract_code_span(out)
					if span is None:
						code_spans.append((0, 0))
						code_texts.append("")
					else:
						code_spans.append(compute_token_span(tokenizer, out, span))
						code_texts.append(out[span[0] : span[1]])
					if cand_tokens:
						full_token_ids.append(list(prompt_token_ids + cand_tokens))
					else:
						full_enc = tokenizer(prompt_text + out, return_tensors="pt")
						full_token_ids.append(full_enc["input_ids"][0].tolist())

				labels = [
					# Set labels later
					False
					for _ in code_texts
				]

				record = {
					"id": problem["task_id"],
					"prompt": prompt_raw,
					"language": "python",
					"difficulty": "unknown",
					"output": output_texts,
					"stripped_output": stripped_outputs,
					"is_correct": labels,
					"code_token_idx": code_spans,
					"program": code_texts,
					"prompt_token_count": prompt_token_count,
					"prompt_token_ids": prompt_token_ids,
					"token_ids": full_token_ids,
				}
				f.write(json.dumps(record) + "\n")
				duration = time.perf_counter() - problem_start
				LOGGER.info("Processed task %s (%d/%d) in %.2fs", problem["task_id"], idx + 1, len(problems), duration)
	finally:
		runner.shutdown()


if __name__ == "__main__":
	main()