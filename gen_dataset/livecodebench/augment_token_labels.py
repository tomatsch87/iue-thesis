"""
Augment LiveCodeBench generations with token-level correctness labels.

Default behavior:
- Reads build-dataset JSONL (`--input-path`) and writes augmented JSONL (`--output-path`).
- For incorrect programs, generates up to `n=4` repair attempts with default fix model
	`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Sampling defaults for fixes: `temperature=1.0`, `top_p=1.0`, `top_k=0`,
	`repetition_penalty=1.0`, `max_tokens=32000`, `max_model_len=32768`.
- Evaluates fix candidates (`--eval-timeout 10`) and aligns buggy vs fixed tokens to build
	`token_labels`, line-level derived flags, and `fixed_program` sentinels.
- Writes a sidecar stats file at `<output-path>.stats.json`.

Important options:
- Dataset scope: `--release-version`, `--full-dataset`, `--start-date`, `--end-date`,
	`--start-row`, `--max-rows`.
- Tokenization/evaluation: `--tokenizer`, `--label-tokenizer`, `--line-filter-threshold`.
- vLLM runtime: `--tensor-parallel-size`, `--gpu-memory-utilization`, `--dtype`,
	`--enable-prefix-caching`, `--trust-remote-code`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .problems import CodeGenerationProblem, load_code_generation_dataset
from .testing_util import run_test

LOGGER = logging.getLogger(__name__)

DEFAULT_FIX_SYSTEM_PROMPT = (
	"You are a code repair assistant. Your task is to fix buggy programs by changing the minimum number of tokens. "
	"Only return the corrected code in a single fenced code block."
)

NO_FIX_FOUND_SENTINEL = "<NO_FIX_FOUND>"
NO_PROGRAM_SENTINEL = "<NO_PROGRAM>"
CORRECT_PROGRAM_SENTINEL = "<CORRECT_PROGRAM>"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Augment LiveCodeBench JSONL with token-level labels and optional fixes",
	)
	parser.add_argument("--input-path", type=Path, required=True, help="Input JSONL from build_dataset.py")
	parser.add_argument("--output-path", type=Path, required=True, help="Output JSONL with token labels")
	parser.add_argument(
		"--model",
		default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
		help="Model name or path usable by vllm for fixing",
	)
	parser.add_argument("--tokenizer", default=None, help="Tokenizer name for the fix model; defaults to --model")
	parser.add_argument(
		"--label-tokenizer",
		default=None,
		help="Tokenizer used to compute token labels; defaults to --tokenizer/--model",
	)
	parser.add_argument("--release-version", default="release_v6", help="LiveCodeBench release version tag")
	parser.add_argument("--full-dataset", action="store_true", help="Use full dataset instead of lite split")
	parser.add_argument("--start-date", default=None, help="Filter contest start date YYYY-MM-DD")
	parser.add_argument("--end-date", default=None, help="Filter contest end date YYYY-MM-DD")
	parser.add_argument("--n", type=int, default=4, help="Number of fixing generations per prompt")
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--top-p", type=float, default=1.0)
	parser.add_argument("--top-k", type=int, default=0)
	parser.add_argument("--repetition-penalty", type=float, default=1.0)
	parser.add_argument("--max-tokens", type=int, default=32000)
	parser.add_argument("--max-model-len", type=int, default=32768)
	parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
	parser.add_argument("--tensor-parallel-size", type=int, default=4)
	parser.add_argument("--dtype", default="bfloat16")
	parser.add_argument("--enable-prefix-caching", action="store_true")
	parser.add_argument("--trust-remote-code", action="store_true")
	parser.add_argument("--eval-timeout", type=int, default=10, help="Timeout per test run (seconds)")
	parser.add_argument("--fix-system-prompt", default=DEFAULT_FIX_SYSTEM_PROMPT)
	parser.add_argument(
		"--max-rows",
		type=int,
		default=None,
		help="Optional limit on number of JSONL rows to process",
	)
	parser.add_argument(
		"--start-row",
		type=int,
		default=0,
		help="Start processing from this input row index (0-based)",
	)
	parser.add_argument(
		"--line-filter-threshold",
		type=int,
		default=4,
		help="Minimum number of incorrect lines to set line_filter=true",
	)
	return parser.parse_args()


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


def evaluate_candidate(problem: CodeGenerationProblem, code: str, timeout: int) -> bool:
	sample = problem.build_eval_sample()
	if not code or not code.strip():
		return False
	try:
		results, metadata = run_test(sample, test=code, debug=False, timeout=timeout)
		error_msg = metadata.get("error_message") if isinstance(metadata, dict) else None
		error_code = metadata.get("error_code") if isinstance(metadata, dict) else None
		if error_msg and ("subprocess crashed" in error_msg or "subprocess timed out" in error_msg):
			LOGGER.warning(
				"Evaluation failure for problem %s: %s (error_code=%s)",
				getattr(problem, "question_id", "unknown"),
				error_msg,
				error_code,
			)
		return bool(results) and all(r is True or r == 1 for r in results)
	except Exception:
		return False


def build_fix_prompt(problem_prompt: str, program: str, system_prompt: str) -> list[dict[str, str]]:
	user_prompt = (
		"Below is a programming problem followed by a buggy solution.\n\n"
		f"{problem_prompt}\n\n"
		"### Buggy solution:\n"
		f"{program}\n\n"
		"Fix the solution by editing the **minimum** number of tokens needed to make the program pass all tests. "
		"Preserve existing comments when they remain correct; update or remove comments if their context is wrong. "
		"Only rename identifiers if they are actually wrong or misleading. "
		"Only return the corrected code in a single fenced code block. "
	)
	return [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": user_prompt},
	]


def compute_token_labels_from_fix(
	buggy_tokens: list[int],
	fixed_tokens: list[int],
	buggy_force_true_mask: Optional[list[bool]] = None,
	fixed_force_true_mask: Optional[list[bool]] = None,
	) -> tuple[list[bool], bool, list[int]]:
	"""Label buggy code tokens via code-only alignment; keep has_equal_match from full-token alignment."""
	labels = [False for _ in buggy_tokens]
	if not buggy_tokens or not fixed_tokens:
		return labels, False, []
	# SequenceMatcher works on sequences; align token ids by equality
	from difflib import SequenceMatcher

	matcher = SequenceMatcher(a=buggy_tokens, b=fixed_tokens, autojunk=False)
	has_equal_match = False
	insertion_indices: list[int] = []

	buggy_mask = buggy_force_true_mask or [False for _ in buggy_tokens]
	fixed_mask = fixed_force_true_mask or [False for _ in fixed_tokens]
	if len(buggy_mask) != len(buggy_tokens):
		buggy_mask = [False for _ in buggy_tokens]
	if len(fixed_mask) != len(fixed_tokens):
		fixed_mask = [False for _ in fixed_tokens]

	for tag, a0, a1, b0, b1 in matcher.get_opcodes():
		if tag == "equal":
			has_equal_match = True
		elif tag == "insert":
			insert_has_code = any(not fixed_mask[fixed_idx] for fixed_idx in range(b0, b1))
			if insert_has_code:
				insertion_indices.append(a0)

	buggy_code_indices = [idx for idx, is_masked in enumerate(buggy_mask) if not is_masked]
	fixed_code_indices = [idx for idx, is_masked in enumerate(fixed_mask) if not is_masked]
	if not buggy_code_indices or not fixed_code_indices:
		return labels, has_equal_match, insertion_indices

	buggy_code_tokens = [buggy_tokens[idx] for idx in buggy_code_indices]
	fixed_code_tokens = [fixed_tokens[idx] for idx in fixed_code_indices]
	code_matcher = SequenceMatcher(a=buggy_code_tokens, b=fixed_code_tokens, autojunk=False)
	for tag, a0, a1, b0, b1 in code_matcher.get_opcodes():
		if tag == "equal":
			for code_idx in range(a0, a1):
				labels[buggy_code_indices[code_idx]] = True
	return labels, has_equal_match, insertion_indices


def resolve_insertion_mismatch_lines(
	insertion_indices: list[int],
	line_numbers: list[int],
	force_true_mask: list[bool],
) -> set[int]:
	if not insertion_indices or not line_numbers:
		return set()
	if force_true_mask and len(force_true_mask) != len(line_numbers):
		force_true_mask = []

	line_has_code: dict[int, bool] = {}
	for idx, line_no in enumerate(line_numbers):
		is_masked = bool(force_true_mask and force_true_mask[idx])
		if line_no not in line_has_code:
			line_has_code[line_no] = False
		if not is_masked:
			line_has_code[line_no] = True

	code_lines = sorted(line_no for line_no, has_code in line_has_code.items() if has_code)
	if not code_lines:
		return set()

	forced_lines: set[int] = set()
	for insert_idx in insertion_indices:
		if insert_idx < 0:
			continue
		if insert_idx < len(line_numbers):
			insertion_line = line_numbers[insert_idx]
		else:
			insertion_line = line_numbers[-1] + 1

		target_line: Optional[int] = None
		if line_has_code.get(insertion_line, False):
			target_line = insertion_line
		else:
			for candidate_line in code_lines:
				if candidate_line > insertion_line:
					target_line = candidate_line
					break
			if target_line is None:
				previous_lines = [line_no for line_no in code_lines if line_no < insertion_line]
				if previous_lines:
					target_line = previous_lines[-1]
				else:
					target_line = code_lines[-1]

		if target_line is not None:
			forced_lines.add(target_line)

	return forced_lines


def iter_jsonl(path: Path) -> Iterable[dict]:
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def decode_token_spans(
	label_tokenizer: PreTrainedTokenizerBase,
	code_response_token_ids: list[int],
) -> tuple[str, list[tuple[int, int]], list[str]]:
	if not code_response_token_ids:
		return "", [], []

	code_text = label_tokenizer.decode(
		code_response_token_ids,
		skip_special_tokens=False,
		clean_up_tokenization_spaces=False,
	)

	try:
		encoded = label_tokenizer(
			code_text,
			add_special_tokens=False,
			return_offsets_mapping=True,
		)
		reencoded_ids = encoded.get("input_ids") or []
		offsets = encoded.get("offset_mapping") or []
		if reencoded_ids == code_response_token_ids and len(offsets) == len(code_response_token_ids):
			spans = [(int(start), int(end)) for start, end in offsets]
			pieces = [code_text[start:end] for start, end in spans]
			return code_text, spans, pieces
	except Exception:
		pass

	decoded_pieces: list[str] = []
	spans: list[tuple[int, int]] = []
	cursor = 0
	for token_id in code_response_token_ids:
		piece = label_tokenizer.decode(
			[token_id],
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)
		decoded_pieces.append(piece)
		piece_len = len(piece)
		spans.append((cursor, cursor + piece_len))
		cursor += piece_len

	code_text = "".join(decoded_pieces)
	return code_text, spans, decoded_pieces


def build_force_true_mask(
	label_tokenizer: PreTrainedTokenizerBase,
	code_response_token_ids: list[int],
) -> list[bool]:
	if not code_response_token_ids:
		return []
	code_text, spans, decoded_pieces = decode_token_spans(label_tokenizer, code_response_token_ids)
	if not code_text:
		return [False for _ in code_response_token_ids]

	comment_spans: list[tuple[int, int]] = []
	for match in re.finditer(r"#.*", code_text):
		comment_spans.append((match.start(), match.end()))

	for match in re.finditer(r"(?m)^[ \t]*[rRuUfF]{0,2}('''|\"\")(?:.|\n)*?\1", code_text):
		comment_spans.append((match.start(), match.end()))

	force_true: list[bool] = []
	for idx, (start, end) in enumerate(spans):
		piece = decoded_pieces[idx]
		is_whitespace = bool(piece) and piece.strip() == ""
		is_comment = any(start < span_end and end > span_start for span_start, span_end in comment_spans)
		force_true.append(is_whitespace or is_comment)
	return force_true


def build_line_numbers(
	label_tokenizer: PreTrainedTokenizerBase,
	code_response_token_ids: list[int],
) -> tuple[str, list[int]]:
	if not code_response_token_ids:
		return "", []
	code_text, spans, _ = decode_token_spans(label_tokenizer, code_response_token_ids)
	line_starts = [0]
	for idx, ch in enumerate(code_text):
		if ch == "\n":
			line_starts.append(idx + 1)

	line_numbers: list[int] = []
	for start, _ in spans:
		line_no = 1
		for line_start in line_starts:
			if line_start <= start:
				line_no += 1
			else:
				break
		line_numbers.append(max(1, line_no - 1))
	return code_text, line_numbers


def apply_line_level_labels(
	labels: list[bool],
	line_numbers: list[int],
	force_true_mask: list[bool],
	forced_incorrect_lines: Optional[set[int]] = None,
) -> tuple[list[bool], int]:
	if not labels or not line_numbers or len(labels) != len(line_numbers):
		return labels, 0
	if force_true_mask and len(force_true_mask) != len(labels):
		force_true_mask = []
	line_has_mismatch: dict[int, bool] = {line_no: True for line_no in (forced_incorrect_lines or set())}
	for idx, label in enumerate(labels):
		is_masked = bool(force_true_mask and force_true_mask[idx])
		if not label and not is_masked:
			line_has_mismatch[line_numbers[idx]] = True
	incorrect_lines = len(line_has_mismatch)
	if incorrect_lines == 0:
		return labels, 0
	line_level = labels[:]
	for idx, line_no in enumerate(line_numbers):
		if line_has_mismatch.get(line_no, False):
			line_level[idx] = False
	if force_true_mask:
		for idx, force_true in enumerate(force_true_mask):
			if force_true:
				line_level[idx] = True
	return line_level, incorrect_lines


def has_any_correct_code_line(
	labels: list[bool],
	line_numbers: list[int],
	force_true_mask: list[bool],
	forced_incorrect_lines: Optional[set[int]] = None,
) -> bool:
	if not labels or not line_numbers or len(labels) != len(line_numbers):
		return False
	if force_true_mask and len(force_true_mask) != len(labels):
		force_true_mask = []
	forced_incorrect_lines = forced_incorrect_lines or set()

	line_summary: dict[int, dict[str, bool]] = {}
	for idx, label in enumerate(labels):
		is_masked = bool(force_true_mask and force_true_mask[idx])
		if is_masked:
			continue
		line_no = line_numbers[idx]
		if line_no not in line_summary:
			line_summary[line_no] = {"has_code": False, "all_true": True}
		line_summary[line_no]["has_code"] = True
		if not label:
			line_summary[line_no]["all_true"] = False

	for line_no in forced_incorrect_lines:
		if line_no in line_summary and line_summary[line_no]["has_code"]:
			line_summary[line_no]["all_true"] = False

	return any(state["has_code"] and state["all_true"] for state in line_summary.values())


def main() -> None:
	from .vllm_runner import VLLMRunner
	import torch

	torch.set_float32_matmul_precision('high')

	args = parse_args()
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	stats_path = args.output_path.with_suffix(args.output_path.suffix + ".stats.json")

	fix_tokenizer_name = args.tokenizer or args.model
	label_tokenizer_name = args.label_tokenizer or fix_tokenizer_name
	fix_tokenizer = AutoTokenizer.from_pretrained(fix_tokenizer_name, trust_remote_code=args.trust_remote_code)
	label_tokenizer = fix_tokenizer
	if label_tokenizer_name != fix_tokenizer_name:
		label_tokenizer = AutoTokenizer.from_pretrained(label_tokenizer_name, trust_remote_code=args.trust_remote_code)

	problems = load_code_generation_dataset(
		release_version=args.release_version,
		start_date=args.start_date,
		end_date=args.end_date,
		use_lite_split=not args.full_dataset,
	)
	problem_map = {p.question_id: p for p in problems}

	runner = VLLMRunner(
		model_name=args.model,
		tokenizer_name=fix_tokenizer_name,
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

	args.output_path.parent.mkdir(parents=True, exist_ok=True)
	stats = {
		"correct_candidates": 0,
		"skipped_no_program": 0,
		"fix_attempted": 0,
		"fix_found": 0,
	}
	try:
		out_mode = "a" if args.output_path.exists() else "w"
		with args.output_path.open(out_mode, encoding="utf-8") as out_f:
			processed_rows = 0
			input_row_idx = 0
			for record in iter_jsonl(args.input_path):
				if input_row_idx < args.start_row:
					input_row_idx += 1
					continue
				if args.max_rows is not None and processed_rows >= args.max_rows:
					break
				problem_id = record.get("id")
				problem = problem_map.get(problem_id)
				if problem is None:
					LOGGER.warning("Skipping unknown problem id: %s", problem_id)
					input_row_idx += 1
					continue

				programs = record.get("program", [])
				is_correct = record.get("is_correct", [])
				code_token_idx = record.get("code_token_idx", [])
				full_token_ids = record.get("token_ids", [])
				prompt_token_count = int(record.get("prompt_token_count", 0))

				token_labels: list[list[dict[str, object]]] = []
				fixed_programs: list[str] = []
				line_filters: list[bool] = []
				no_correct_line_filters: list[bool] = []

				for idx, correct in enumerate(is_correct):
					program = programs[idx] if idx < len(programs) else ""
					candidate_token_ids = full_token_ids[idx] if idx < len(full_token_ids) else []
					response_token_ids = candidate_token_ids[prompt_token_count:]
					span = code_token_idx[idx] if idx < len(code_token_idx) else (0, 0)
					if not isinstance(span, (list, tuple)) or len(span) != 2:
						span = (0, 0)
					span_start, span_end = int(span[0]), int(span[1])
					force_no_program = span_start == 0 and span_end == 0
					if span_start < 0 or span_end < span_start or force_no_program:
						code_response_token_ids: list[int] = []
					else:
						code_response_token_ids = response_token_ids[span_start : span_end + 1]

					token_texts: list[str] = []
					force_true_mask: list[bool] = []
					code_text = ""
					line_numbers: list[int] = []
					if code_response_token_ids:
						token_texts = label_tokenizer.convert_ids_to_tokens(code_response_token_ids)
						if len(token_texts) != len(code_response_token_ids):
							token_texts = [str(token_id) for token_id in code_response_token_ids]
						code_text, line_numbers = build_line_numbers(label_tokenizer, code_response_token_ids)
						if len(line_numbers) != len(code_response_token_ids):
							line_numbers = [1 for _ in code_response_token_ids]

						force_true_mask = build_force_true_mask(label_tokenizer, code_response_token_ids)
						if len(force_true_mask) != len(code_response_token_ids):
							force_true_mask = []

					if correct is True:
						stats["correct_candidates"] += 1
						token_labels.append(
							[
								{
									"token_id": token_id,
									"token_text": token_texts[idx]
									if idx < len(token_texts) and token_texts[idx] is not None
									else str(token_id),
									"label": True,
									"line": line_numbers[idx] if idx < len(line_numbers) else 1,
									"ws_comment": force_true_mask[idx]
									if force_true_mask and idx < len(force_true_mask)
									else False,
								}
								for idx, token_id in enumerate(code_response_token_ids)
							]
						)
						fixed_programs.append(CORRECT_PROGRAM_SENTINEL)
						line_filters.append(False)
						no_correct_line_filters.append(False)
						continue

					no_program = force_no_program or not program or not program.strip()
					if no_program:
						stats["skipped_no_program"] += 1
						token_labels.append([])
						fixed_programs.append(NO_PROGRAM_SENTINEL)
						line_filters.append(False)
						no_correct_line_filters.append(False)
						continue

					stats["fix_attempted"] += 1
					LOGGER.info("Attempting fix for row %s (problem id: %s)", processed_rows, problem_id)

					messages = build_fix_prompt(record.get("prompt", ""), program, args.fix_system_prompt)
					try:
						prompt_text = fix_tokenizer.apply_chat_template(
							messages,
							tokenize=False,
							add_generation_prompt=True,
						)
					except Exception:
						prompt_text = "\n\n".join(m["content"] for m in messages if m.get("content"))

					gens = runner.generate([prompt_text])
					candidates = gens[0] if gens else []
					fixed_program = ""
					fixed_token_ids: list[int] = []

					for cand in candidates:
						cand_tokens = cand.get("token_ids") or []
						decoded = fix_tokenizer.decode(cand_tokens, skip_special_tokens=False) if cand_tokens else cand.get("text", "")
						span = extract_code_span(decoded)
						if span is None:
							continue
						code_text = decoded[span[0] : span[1]]
						if not code_text.strip():
							continue
						if evaluate_candidate(problem, code_text, timeout=args.eval_timeout):
							fixed_program = code_text
							fixed_token_ids = label_tokenizer(code_text, add_special_tokens=False)["input_ids"]
							break

					if fixed_program:
						stats["fix_found"] += 1
						fixed_force_true_mask = build_force_true_mask(label_tokenizer, fixed_token_ids)
						if len(fixed_force_true_mask) != len(fixed_token_ids):
							fixed_force_true_mask = []
						labels, _, insertion_indices = compute_token_labels_from_fix(
							code_response_token_ids,
							fixed_token_ids,
							buggy_force_true_mask=force_true_mask,
							fixed_force_true_mask=fixed_force_true_mask,
						)
						forced_incorrect_lines = resolve_insertion_mismatch_lines(
							insertion_indices,
							line_numbers,
							force_true_mask,
						)
						if force_true_mask:
							for label_idx, force_true in enumerate(force_true_mask):
								if force_true:
									labels[label_idx] = True
						labels, incorrect_lines = apply_line_level_labels(
							labels,
							line_numbers,
							force_true_mask,
							forced_incorrect_lines=forced_incorrect_lines,
						)
						has_correct_code_line = has_any_correct_code_line(
							labels,
							line_numbers,
							force_true_mask,
							forced_incorrect_lines=forced_incorrect_lines,
						)
						line_filters.append(incorrect_lines >= args.line_filter_threshold)
						no_correct_line_filters.append(not has_correct_code_line)
						token_labels.append(
							[
								{
									"token_id": token_id,
									"token_text": token_texts[idx]
									if idx < len(token_texts) and token_texts[idx] is not None
									else str(token_id),
									"label": labels[idx],
									"line": line_numbers[idx] if idx < len(line_numbers) else 1,
									"ws_comment": force_true_mask[idx]
									if force_true_mask and idx < len(force_true_mask)
									else False,
								}
								for idx, token_id in enumerate(code_response_token_ids)
							]
						)
						fixed_programs.append(fixed_program)
					else:
						labels = [False for _ in code_response_token_ids]
						if force_true_mask:
							for label_idx, force_true in enumerate(force_true_mask):
								if force_true:
									labels[label_idx] = True
						line_filters.append(False)
						no_correct_line_filters.append(False)
						token_labels.append(
							[
								{
									"token_id": token_id,
									"token_text": token_texts[idx]
									if idx < len(token_texts) and token_texts[idx] is not None
									else str(token_id),
									"label": labels[idx],
									"line": line_numbers[idx] if idx < len(line_numbers) else 1,
									"ws_comment": force_true_mask[idx]
									if force_true_mask and idx < len(force_true_mask)
									else False,
								}
								for idx, token_id in enumerate(code_response_token_ids)
							]
						)
						fixed_programs.append(NO_FIX_FOUND_SENTINEL)

				record["token_labels"] = token_labels
				record["fixed_program"] = fixed_programs
				record["line_filter"] = line_filters
				record["no_correct_line_filter"] = no_correct_line_filters
				out_f.write(json.dumps(record) + "\n")
				processed_rows += 1
				input_row_idx += 1
		with stats_path.open("w", encoding="utf-8") as stats_f:
			stats_f.write(json.dumps(stats, indent=2) + "\n")
	finally:
		runner.shutdown()


if __name__ == "__main__":
	main()