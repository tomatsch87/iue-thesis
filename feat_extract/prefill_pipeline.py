"""
Extract per-candidate hidden-state (and optionally attention) features from prompt+response tokens.

Default behavior:
- Loads model `Qwen/Qwen3-Coder-30B-A3B-Instruct` and dataset
	`livecodebench/livecodebench_qwen3`.
- Processes splits `train`, `validation`, `test` unless overridden.
- Extracts both hidden states (attentions) by default for token position `0`
	(relative to assistant tokens).
- Layer defaults are `None`, which means all available model layers are used.
- Saves one `.pt` file per sample under `--output-dir` (default
	`/data/feats_lcb_qwen3_code_segment`) with per-candidate feature payloads.
- Uses pre-tokenized fields (`prompt_token_count`, `token_ids`) when available,
	otherwise tokenizes prompts/candidates on the fly.

Important options:
- Data source: `--input-jsonl-map` (highest priority), `--input-jsonl`, or
	Hugging Face `--dataset-name/--dataset-config`.
- Target controls: `--hidden-state-layers`, `--hidden-state-token-positions`,
	`--attention-layers`, `--attention-token-positions`,
	`--disable-hidden-states`, `--disable-attentions`.
- Token specifiers support integers, negative tail indices, `all`,
	`first_code_token`, `last_code_token`, and `code_to_end`.
- Optional diagnostics/features: `--include-token-entropy`,
	`--token-entropy-top-k`, `--include-logtoku`, `--logtoku-top-l`.
- Throughput/scope: `--max-samples-per-split`, `--max-outputs-per-sample`,
	`--output-field`, `--attn-implementation`.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, TypeVar, Union
import gc

import torch
import torch.nn.functional as F
from torch.special import digamma
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
	from global_utils import seed_everything
except ModuleNotFoundError:
	from utils import seed_everything


LOGGER = logging.getLogger(__name__)

# local default variables
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_DATASET_NAME = "livecodebench"
DEFAULT_DATASET_CONFIG = "livecodebench_qwen3"
DEFAULT_OUTPUT_DIR = "/data/feats_lcb_qwen3_code_segment"

TokenSpecifier = Union[int, str]
CODE_TOKEN_SPECIFIERS = {"first_code_token", "last_code_token"}
CODE_TO_END_SPECIFIER = "code_to_end"
_T = TypeVar("_T", bound=Hashable)


@dataclass
class ExtractionTargets:
	hidden_state_layers: Optional[List[int]]
	hidden_state_tokens: List[TokenSpecifier]
	attention_layers: Optional[List[int]]
	attention_tokens: List[TokenSpecifier]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Prefill feature extraction pipeline")
	parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model identifier")
	parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Dataset name")
	parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG, help="Dataset configuration")
	parser.add_argument(
		"--input-jsonl",
		type=str,
		default=None,
		help="Optional local JSONL bypasses dataset-name/config.",
	)
	parser.add_argument(
		"--input-jsonl-map",
		nargs="*",
		help=(
			"Optional split-to-path mapping for JSONL inputs, format split=path (e.g. train=... val=...). "
			"If provided, overrides --input-jsonl."
		),
	)
	parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where extracted features will be stored")
	parser.add_argument(
		"--splits",
		nargs="*",
		default=("train", "validation", "test"),
		help="Dataset splits to process",
	)
	parser.add_argument(
		"--hidden-state-layers",
		nargs="*",
		type=int,
		default=None,
		help="Transformer layer indices for hidden state extraction (embedding layer is 0).",
	)
	parser.add_argument(
		"--hidden-state-token-positions",
		nargs="*",
		type=str,
		default=None,
		help=(
			"Token positions (0-based) relative to assistant tokens for hidden state extraction. "
			"Use negative numbers (e.g. -1) to index from the end, 'all' to include every assistant token, "
			"'first_code_token'/'last_code_token' to target the code span, and 'code_to_end' to span "
			"from the first code token through the final assistant token when available."
		),
	)
	parser.add_argument(
		"--attention-layers",
		nargs="*",
		type=int,
		default=None,
		help="Transformer layer indices for attention extraction.",
	)
	parser.add_argument(
		"--attention-token-positions",
		nargs="*",
		type=str,
		default=None,
		help=(
			"Token positions (0-based) relative to assistant tokens for attention extraction. "
			"Use negative numbers (e.g. -1) to index from the end, 'all' to include every assistant token, "
			"'first_code_token'/'last_code_token' to target the code span, and 'code_to_end' to span "
			"from the first code token through the final assistant token when available."
		),
	)
	parser.add_argument(
		"--max-outputs-per-sample",
		type=int,
		default=None,
		help="Limit how many candidate outputs to process per sample.",
	)
	parser.add_argument(
		"--max-samples-per-split",
		type=int,
		default=None,
		help="Limit number of samples processed per split.",
	)
	parser.add_argument(
		"--disable-hidden-states",
		action="store_true",
		help="Skip hidden state extraction entirely.",
	)
	parser.add_argument(
		"--disable-attentions",
		action="store_true",
		help="Skip attention extraction entirely.",
	)
	parser.add_argument(
		"--attn-implementation",
		default="eager",
		help="Attention implementation to set on the model.",
	)
	parser.add_argument(
		"--system-prompt",
		type=str,
		default=None,
		help="Optional system prompt injected ahead of the user message (matches GPT-OSS behavior when provided).",
	)
	parser.add_argument(
		"--enable-reasoning-traces",
		action="store_true",
		help="Split assistant responses into reasoning and final content using --reasoning-trace-delimiter.",
	)
	parser.add_argument(
		"--reasoning-trace-delimiter",
		type=str,
		default=None,
		help="Delimiter used to separate reasoning traces from final content when reasoning traces are enabled.",
	)
	parser.add_argument(
		"--include-token-entropy",
		action="store_true",
		help="Calculate token entropy (nats) for hidden-state token positions and store alongside extracted features.",
	)
	parser.add_argument(
		"--include-logtoku",
		action="store_true",
		help="Calculate LogTokU reliability for hidden-state token positions and store alongside extracted features.",
	)
	parser.add_argument(
		"--output-field",
		type=str,
		default="output",
		help=(
			"Dataset field to read assistant candidates from. "
			"Defaults to 'output' and will fall back to 'stripped_output' if missing."
		),
	)
	parser.add_argument(
		"--token-entropy-top-k",
		type=int,
		default=None,
		help=(
			"Optional cap on logits considered when computing token entropy. "
			"Remaining probability mass is treated as a single tail bucket."
		),
	)
	parser.add_argument(
		"--logtoku-top-l",
		type=int,
		default=50,
		help="Number of top logits to use as evidence when computing LogTokU (L in the paper).",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		help="Logging level (e.g. INFO, DEBUG).",
	)
	return parser.parse_args()


def setup_logging(level: str) -> None:
	logging.basicConfig(
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		level=getattr(logging, level.upper(), logging.INFO),
	)


def unique_preserve_order(values: Sequence[_T]) -> List[_T]:
	seen: set[_T] = set()
	ordered: List[_T] = []
	for val in values:
		if val not in seen:
			seen.add(val)
			ordered.append(val)
	return ordered


def select_outputs_field(sample: Dict[str, object], preferred_field: Optional[str]) -> Optional[object]:
	fields = [preferred_field] if preferred_field else []
	fields.append("output")
	fields.append("stripped_output")
	for key in unique_preserve_order([f for f in fields if f]):
		if key in sample and sample[key] is not None:
			return sample[key]
	return None


def parse_jsonl_mapping(values: Optional[Sequence[str]]) -> Dict[str, str]:
	mapping: Dict[str, str] = {}
	if not values:
		return mapping
	for raw in values:
		if not raw:
			continue
		delims = ["=", ":"]
		split_name = None
		path = None
		for delim in delims:
			if delim in raw:
				parts = raw.split(delim, 1)
				if len(parts) == 2:
					split_name, path = parts[0].strip(), parts[1].strip()
				break
		if not split_name or not path:
			LOGGER.warning("Ignoring invalid input-jsonl-map entry '%s' (expected split=path)", raw)
			continue
		resolved = str(Path(path).expanduser())
		mapping[split_name] = resolved
	return mapping


def parse_token_specifiers(
	values: Optional[Sequence[str]],
	label: str,
	default: Optional[Sequence[TokenSpecifier]] = None,
) -> List[TokenSpecifier]:
	if values is None:
		return list(default) if default is not None else []

	parsed: List[TokenSpecifier] = []
	for raw in values:
		if raw is None:
			continue
		candidate = raw.strip()
		if candidate == "":
			continue
		normalized = candidate.lower()
		if normalized == "all":
			parsed.append("all")
			continue
		if normalized == CODE_TO_END_SPECIFIER:
			parsed.append(normalized)
			continue
		if normalized in CODE_TOKEN_SPECIFIERS:
			parsed.append(normalized)
			continue

		try:
			parsed.append(int(candidate))
		except ValueError:
			LOGGER.warning("Unrecognized %s token specifier '%s'; skipping", label, candidate)
			continue
	return unique_preserve_order(parsed)


def resolve_token_positions(
	specifiers: Sequence[TokenSpecifier],
	assistant_token_count: int,
	label: str,
	code_token_range: Optional[Tuple[int, int]] = None,
) -> List[int]:
	if not specifiers:
		return []

	resolved: List[int] = []
	for spec in specifiers:
		if isinstance(spec, str):
			normalized = spec.lower()
			if normalized == "all":
				resolved.extend(range(assistant_token_count))
				continue
			if normalized == CODE_TO_END_SPECIFIER or normalized in CODE_TOKEN_SPECIFIERS:
				if code_token_range is None:
					continue
				code_start_raw, code_end_raw = code_token_range
				try:
					code_start = int(code_start_raw)
					code_end = int(code_end_raw)
				except (TypeError, ValueError):
					LOGGER.warning(
						"Invalid code token range %s for %s; skipping",
						code_token_range,
						label,
					)
					continue
				if code_start == 0 and code_end == 0:
					continue
				if code_end < code_start:
					LOGGER.warning(
						"Ignoring code token range %s for %s (end before start)",
						code_token_range,
						label,
					)
					continue
				if normalized == CODE_TO_END_SPECIFIER:
					if assistant_token_count <= 0:
						LOGGER.warning(
							"Cannot apply code_to_end for %s without assistant tokens",
							label,
						)
						continue
					start_idx = max(code_start, 0)
					if start_idx >= assistant_token_count:
						LOGGER.warning(
							"code_to_end start %s outside assistant token range [0, %s) for %s",
							start_idx,
							assistant_token_count,
							label,
						)
						continue
					resolved.extend(range(start_idx, assistant_token_count))
				elif normalized == "first_code_token":
					resolved.append(code_start)
				else:
					resolved.append(code_end)
				continue
			LOGGER.warning("Unknown %s token specifier '%s'; skipping", label, spec)
			continue
		else:
			rel_pos = spec
			if rel_pos >= 0:
				resolved.append(rel_pos)
				continue
			if assistant_token_count <= 0:
				LOGGER.warning(
					"Cannot resolve negative %s token position %s without assistant tokens", label, rel_pos
				)
				continue
			target_index = assistant_token_count + rel_pos
			if target_index < 0:
				LOGGER.warning(
					"Ignoring %s token position %s (resolved to %s) outside range [0, %s)",
					label,
					rel_pos,
					target_index,
					assistant_token_count,
				)
				continue
			resolved.append(target_index)

	filtered: List[int] = []
	for pos in unique_preserve_order(resolved):
		if pos < 0 or pos >= assistant_token_count:
			LOGGER.warning(
				"Ignoring %s token position %s outside range [0, %s)", label, pos, assistant_token_count
			)
			continue
		filtered.append(pos)
	return filtered


def format_token_specifier(spec: TokenSpecifier) -> str:
	if isinstance(spec, str):
		return spec.replace("/", "_").lower()
	if spec >= 0:
		return str(spec)
	return f"neg{abs(spec)}"


def configure_extraction_targets(args: argparse.Namespace) -> ExtractionTargets:
	hidden_tokens = parse_token_specifiers(
		args.hidden_state_token_positions,
		label="hidden state",
		default=[0],
	)
	attention_tokens = parse_token_specifiers(
		args.attention_token_positions,
		label="attention",
		default=[0],
	)

	hidden_layers = unique_preserve_order(args.hidden_state_layers) if args.hidden_state_layers else None
	attention_layers = unique_preserve_order(args.attention_layers) if args.attention_layers else None

	if args.disable_hidden_states:
		hidden_layers = []
		hidden_tokens = []
	if args.disable_attentions:
		attention_layers = []
		attention_tokens = []

	return ExtractionTargets(
		hidden_state_layers=hidden_layers,
		hidden_state_tokens=unique_preserve_order(hidden_tokens),
		attention_layers=attention_layers,
		attention_tokens=unique_preserve_order(attention_tokens),
	)


def build_chat_text(tokenizer, add_generation_prompt: bool, continue_final_message: bool, messages: Sequence[Dict[str, str]]) -> str:
	text = tokenizer.apply_chat_template(
		messages,
		tokenize=False,
		add_generation_prompt=add_generation_prompt,
		continue_final_message=continue_final_message,
	)
	return text


def split_reasoning_trace(candidate: str, delimiter: str) -> Tuple[str, str]:
	if delimiter not in candidate:
		raise ValueError("delimiter not found in candidate")
	parts = candidate.split(delimiter)
	if len(parts) < 2:
		raise ValueError("delimiter split did not yield reasoning and final content")
	reasoning_trace = parts[0].strip()
	final_content = parts[-1].strip()
	return reasoning_trace, final_content


def determine_prompt_length(tokenizer, prompt: str, system_prompt: Optional[str] = None) -> Tuple[int, torch.Tensor]:
	prompt_messages: List[Dict[str, str]] = []
	if system_prompt:
		prompt_messages.append({"role": "system", "content": system_prompt})
	prompt_messages.append({"role": "user", "content": prompt})
	prompt_text = build_chat_text(tokenizer, add_generation_prompt=True, continue_final_message=False, messages=prompt_messages)
	prompt_inputs = tokenizer([prompt_text], return_tensors="pt")
	return prompt_inputs["input_ids"].shape[-1], prompt_inputs["input_ids"][0]


def filter_indices(indices: Optional[Sequence[int]], upper: int, label: str) -> List[int]:
	if indices is None:
		return list(range(upper))
	valid: List[int] = []
	for idx in indices:
		if idx < 0 or idx >= upper:
			LOGGER.warning("Ignoring %s index %s outside valid range [0, %s)", label, idx, upper)
			continue
		valid.append(idx)
	return valid


def clear_cuda_cache() -> None:
	gc.collect()
	torch.cuda.synchronize()
	torch.cuda.empty_cache()


def gather_logits_for_positions(
	logits: torch.Tensor,
	abs_positions: Sequence[int],
) -> torch.Tensor:
	if not abs_positions:
		return torch.empty(0, logits.shape[-1], dtype=logits.dtype, device=logits.device)
	gather_indices = torch.tensor(
		[max(pos - 1, 0) for pos in abs_positions],
		device=logits.device,
		dtype=torch.long,
	)
	return logits.index_select(0, gather_indices)


def compute_token_entropy(logits_slice: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
	if logits_slice.numel() == 0:
		return torch.empty(0, device=logits_slice.device)
	if top_k is None or top_k <= 0 or top_k >= logits_slice.shape[-1]:
		log_probs = F.log_softmax(logits_slice, dim=-1)
		probs = torch.exp(log_probs)
		return -(probs * log_probs).sum(dim=-1)
	logsumexp = torch.logsumexp(logits_slice, dim=-1, keepdim=True)
	k = min(top_k, logits_slice.shape[-1])
	topk_vals, _ = torch.topk(logits_slice, k, dim=-1)
	topk_log_probs = topk_vals - logsumexp
	topk_probs = torch.exp(topk_log_probs)
	tail_prob = torch.clamp(1.0 - topk_probs.sum(dim=-1), min=0.0)
	entropy = -(topk_probs * topk_log_probs).sum(dim=-1)
	positive_tail = tail_prob > 0
	if positive_tail.any():
		tail_log_prob = torch.zeros_like(tail_prob)
		tail_log_prob[positive_tail] = torch.log(tail_prob[positive_tail])
		entropy = entropy - tail_prob * tail_log_prob
	return entropy


def compute_logtoku_reliability(logits_slice: torch.Tensor, top_l: Optional[int]) -> torch.Tensor:
    if logits_slice.numel() == 0:
        return torch.empty(0, device=logits_slice.device)
    if top_l is None or top_l <= 0:
        top_l = logits_slice.shape[-1]
    k = min(top_l, logits_slice.shape[-1])
    topk_vals, _ = torch.topk(logits_slice, k, dim=-1)
    evidence = torch.relu(topk_vals)
    alpha0 = evidence.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    au_term = -(evidence / alpha0) * (digamma(evidence + 1.0) - digamma(alpha0 + 1.0))
    au = au_term.sum(dim=-1)
    eu = float(k) / (evidence + 1.0).sum(dim=-1).clamp_min(1e-9)
    reliability = -au * eu
    return reliability


def extract_for_candidate(
	tokenizer,
	model,
	prompt: str,
	candidate: str,
	prompt_token_count: int,
	targets: ExtractionTargets,
	code_token_range: Optional[Tuple[int, int]] = None,
	system_prompt: Optional[str] = None,
	reasoning_traces_enabled: bool = False,
	reasoning_trace_delimiter: Optional[str] = None,
	include_token_entropy: bool = False,
	token_entropy_top_k: Optional[int] = None,
	include_logtoku: bool = False,
	logtoku_top_l: Optional[int] = None,
	pretok_ids: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
	if pretok_ids is None:
		messages: List[Dict[str, str]] = []
		if system_prompt:
			messages.append({"role": "system", "content": system_prompt})
		messages.append({"role": "user", "content": prompt})
		use_reasoning_trace = reasoning_traces_enabled and bool(reasoning_trace_delimiter)
		if use_reasoning_trace and reasoning_trace_delimiter is not None:
			try:
				reasoning_trace, final_content = split_reasoning_trace(candidate, reasoning_trace_delimiter)
				messages.append({"role": "assistant", "thinking": reasoning_trace, "content": final_content})
			except ValueError:
				LOGGER.warning(
					"Reasoning trace delimiter '%s' not found; falling back to full candidate content.",
					reasoning_trace_delimiter,
				)
				messages.append({"role": "assistant", "content": candidate})
		else:
			messages.append({"role": "assistant", "content": candidate})
		conversation_text = build_chat_text(tokenizer, add_generation_prompt=False, continue_final_message=False, messages=messages)
		model_inputs = tokenizer([conversation_text], return_tensors="pt").to(model.device)
		input_ids_tensor = model_inputs["input_ids"][0]
	else:
		input_ids_tensor = torch.as_tensor(list(pretok_ids), dtype=torch.long, device=model.device)
		attention = torch.ones_like(input_ids_tensor)
		model_inputs = {"input_ids": input_ids_tensor.unsqueeze(0), "attention_mask": attention.unsqueeze(0)}
	input_ids_list = input_ids_tensor.detach().cpu().tolist()

	output = None
	try:
		with torch.inference_mode():
			output = model(
				**model_inputs,
				output_hidden_states=bool(targets.hidden_state_tokens and targets.hidden_state_layers != []),
				output_attentions=bool(targets.attention_tokens and targets.attention_layers != []),
				use_cache=False # No KV cache to get all hidden states/attentions for every decoding step
			)

		sequence_length = model_inputs["input_ids"].shape[-1]
		assistant_token_count = sequence_length - prompt_token_count
		if assistant_token_count <= 0:
			LOGGER.warning("Assistant token count non-positive, skipping candidate")
			result = {
				"assistant_token_count": assistant_token_count,
				"hidden_states": {},
				"attentions": {},
				"token_ids": model_inputs["input_ids"][0].detach().cpu(),
				"resolved_hidden_state_tokens": [],
				"resolved_attention_tokens": [],
			}
			return result

		hidden_token_positions = resolve_token_positions(
			targets.hidden_state_tokens,
			assistant_token_count,
			label="hidden state",
			code_token_range=code_token_range,
		) if targets.hidden_state_tokens else []
		attention_token_positions = resolve_token_positions(
			targets.attention_tokens,
			assistant_token_count,
			label="attention",
			code_token_range=code_token_range,
		) if targets.attention_tokens else []

		resolved_hidden_positions: List[Tuple[int, int]] = []
		if hidden_token_positions:
			for rel_pos in hidden_token_positions:
				abs_pos = prompt_token_count + rel_pos
				if abs_pos >= sequence_length:
					LOGGER.warning("Hidden state token index %s out of range (%s)", abs_pos, sequence_length)
					continue
				resolved_hidden_positions.append((rel_pos, abs_pos))
		hidden_token_pairs: List[Tuple[int, str]] = []
		if resolved_hidden_positions:
			hidden_token_ids = [input_ids_list[abs_pos] for _, abs_pos in resolved_hidden_positions]
			hidden_token_texts = tokenizer.convert_ids_to_tokens(hidden_token_ids)
			hidden_token_pairs = [
				(rel_pos, token_text)
				for (rel_pos, _), token_text in zip(resolved_hidden_positions, hidden_token_texts)
			]

		resolved_attention_positions: List[Tuple[int, int]] = []
		if attention_token_positions:
			for rel_pos in attention_token_positions:
				abs_pos = prompt_token_count + rel_pos
				if abs_pos >= sequence_length:
					LOGGER.warning("Attention token index %s out of range (%s)", abs_pos, sequence_length)
					continue
				resolved_attention_positions.append((rel_pos, abs_pos))
		attention_token_pairs: List[Tuple[int, str]] = []
		if resolved_attention_positions:
			attention_token_ids = [input_ids_list[abs_pos] for _, abs_pos in resolved_attention_positions]
			attention_token_texts = tokenizer.convert_ids_to_tokens(attention_token_ids)
			attention_token_pairs = [
				(rel_pos, token_text)
				for (rel_pos, _), token_text in zip(resolved_attention_positions, attention_token_texts)
			]

		hidden_state_features: Dict[int, Dict[int, torch.Tensor]] = {}
		if output.hidden_states is not None and targets.hidden_state_layers != [] and resolved_hidden_positions:
			available_layers = len(output.hidden_states)
			layers = filter_indices(targets.hidden_state_layers, available_layers, "hidden state layer")
			for layer_idx in layers:
				layer_tensor = output.hidden_states[layer_idx][0]  # (seq_len, hidden_size)
				for rel_pos, abs_pos in resolved_hidden_positions:
					hidden_state_features.setdefault(layer_idx, {})[rel_pos] = (
						layer_tensor[abs_pos].detach().cpu()
					)

		token_entropy_map: Dict[int, float] = {}
		if include_token_entropy and resolved_hidden_positions:
			logits_tensor = output.logits[0]
			abs_positions = [abs_pos for _, abs_pos in resolved_hidden_positions]
			logits_slice = gather_logits_for_positions(logits_tensor, abs_positions).detach()
			if logits_slice.numel():
				entropies = compute_token_entropy(logits_slice, top_k=token_entropy_top_k).cpu().tolist()
				for (rel_pos, _), entropy_value in zip(resolved_hidden_positions, entropies):
					token_entropy_map[rel_pos] = float(entropy_value)

		logtoku_reliability_map: Dict[int, float] = {}
		if include_logtoku and resolved_hidden_positions:
			logits_tensor = output.logits[0]
			abs_positions = [abs_pos for _, abs_pos in resolved_hidden_positions]
			logits_slice = gather_logits_for_positions(logits_tensor, abs_positions).detach()
			if logits_slice.numel():
				reliabilities = compute_logtoku_reliability(logits_slice, top_l=logtoku_top_l).cpu().tolist()
				for (rel_pos, _), reliability_value in zip(resolved_hidden_positions, reliabilities):
					logtoku_reliability_map[rel_pos] = float(reliability_value)

		attention_features: Dict[int, Dict[int, torch.Tensor]] = {}
		if output.attentions is not None and targets.attention_layers != [] and resolved_attention_positions:
			available_layers = len(output.attentions)
			layers = filter_indices(targets.attention_layers, available_layers, "attention layer")
			for layer_idx in layers:
				layer_tensor = output.attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)
				for rel_pos, abs_pos in resolved_attention_positions:
					attention_features.setdefault(layer_idx, {})[rel_pos] = (
						layer_tensor[:, :abs_pos, :abs_pos].detach().cpu()
					)

		result = {
			"assistant_token_count": assistant_token_count,
			"hidden_states": hidden_state_features,
			"attentions": attention_features,
			"token_entropies": token_entropy_map,
			"logtoku_reliability": logtoku_reliability_map,
			"token_ids": input_ids_tensor.detach().cpu(),
			"resolved_hidden_state_tokens": hidden_token_pairs,
			"resolved_attention_tokens": attention_token_pairs,
		}
		return result
	finally:
		if output is not None:
			del output
		del model_inputs
		clear_cuda_cache()


def save_sample_features(
	output_dir: Path,
	split_name: str,
	sample_id: str,
	targets: ExtractionTargets,
	payload: Dict[str, object],
) -> Path:
	# Some datasets (e.g., BigCodeBench) include forward slashes in sample IDs; replace to avoid nested paths
	sanitized_sample_id = sample_id.replace("/", "_")
	target_parts: List[str] = []

	if targets.hidden_state_layers and len(targets.hidden_state_layers) > 0:
		target_parts.append("hsL" + "-".join(str(x) for x in targets.hidden_state_layers))
	elif targets.hidden_state_layers == []:
		target_parts.append("hsLnone")
	else:
		target_parts.append("hsLall")

	if targets.hidden_state_tokens:
		target_parts.append(
			"hsT" + "-".join(format_token_specifier(x) for x in targets.hidden_state_tokens)
		)
	else:
		target_parts.append("hsTnone")

	if targets.attention_layers and len(targets.attention_layers) > 0:
		target_parts.append("attL" + "-".join(str(x) for x in targets.attention_layers))
	elif targets.attention_layers == []:
		target_parts.append("attLnone")
	else:
		target_parts.append("attLall")

	if targets.attention_tokens:
		target_parts.append(
			"attT" + "-".join(format_token_specifier(x) for x in targets.attention_tokens)
		)
	else:
		target_parts.append("attTnone")

	descriptor = "__".join(target_parts)
	filename = f"{sanitized_sample_id}__{descriptor}.pt"
	split_dir = output_dir / split_name
	split_dir.mkdir(parents=True, exist_ok=True)
	path = split_dir / filename
	torch.save(payload, path)
	return path


def main() -> None:
	args = parse_args()
	setup_logging(args.log_level)

	seed_everything(0)

	targets = configure_extraction_targets(args)
	if not targets.hidden_state_tokens and not targets.attention_tokens:
		LOGGER.error("No extraction targets specified. Enable hidden states or attentions.")
		return

	LOGGER.info("Loading tokenizer %s", args.model_name)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name)

	LOGGER.info("Loading model %s", args.model_name)
	model = AutoModelForCausalLM.from_pretrained(
		args.model_name,
		device_map="auto",
		dtype="auto",
		trust_remote_code=True,
	)
	try:
		model.set_attn_implementation(args.attn_implementation)
	except AttributeError:
		LOGGER.warning("Model does not support set_attn_implementation; continuing without change")
	model.eval()

	jsonl_map = parse_jsonl_mapping(args.input_jsonl_map)
	if jsonl_map:
		LOGGER.info("Loading dataset from JSONL mapping %s", jsonl_map)
		dataset = load_dataset("json", data_files=jsonl_map)
	elif args.input_jsonl:
		jsonl_path = Path(args.input_jsonl).expanduser().resolve()
		if not jsonl_path.exists():
			LOGGER.error("Input JSONL %s not found; aborting", jsonl_path)
			return
		LOGGER.info("Loading dataset from JSONL %s", jsonl_path)
		dataset = load_dataset("json", data_files=str(jsonl_path))
	else:
		LOGGER.info("Loading dataset %s (%s)", args.dataset_name, args.dataset_config)
		dataset = load_dataset(args.dataset_name, args.dataset_config)

	output_dir = Path(args.output_dir).expanduser().resolve()
	LOGGER.info("Saving features to %s", output_dir)
	system_prompt = args.system_prompt
	reasoning_traces_enabled = args.enable_reasoning_traces
	reasoning_trace_delimiter = args.reasoning_trace_delimiter
	if reasoning_traces_enabled and not reasoning_trace_delimiter:
		LOGGER.warning("Reasoning traces enabled without delimiter; disabling reasoning trace handling.")
		reasoning_traces_enabled = False

	available_splits = list(dataset.keys()) # type: ignore
	default_split_tuple = ("train", "validation", "test")
	if (args.input_jsonl or jsonl_map) and tuple(args.splits) == default_split_tuple:
		splits_to_process = available_splits
	else:
		splits_to_process = list(args.splits)

	for split_name in splits_to_process:
		if split_name not in dataset:
			LOGGER.warning("Split %s not present in dataset; skipping", split_name)
			continue

		LOGGER.info("Processing split %s", split_name)
		split = dataset[split_name] # type: ignore
		for idx, example in enumerate(split):
			if args.max_samples_per_split is not None and idx >= args.max_samples_per_split:
				LOGGER.info("Reached sample limit (%s) for split %s", args.max_samples_per_split, split_name)
				break

			sample = dict(example)
			sample_id = str(sample.get("id", f"{split_name}_{idx}"))
			prompt = sample.get("prompt")
			outputs = select_outputs_field(sample, args.output_field)
			language = sample.get("language", "unknown")
			is_correct = sample.get("is_correct", [])
			difficulty = sample.get("difficulty", "unknown")
			code_token_idx = sample.get("code_token_idx", [])

			if prompt is None or outputs is None:
				LOGGER.warning("Skipping sample %s due to missing prompt/output", sample_id)
				continue

			candidate_outputs: List[str]
			if isinstance(outputs, str):
				candidate_outputs = [outputs]
			elif isinstance(outputs, Sequence):
				candidate_outputs = [str(c) for c in outputs if isinstance(c, str)]
			else:
				LOGGER.warning("Skipping sample %s due to unsupported outputs type %s", sample_id, type(outputs))
				continue

			if args.max_outputs_per_sample is not None:
				candidate_outputs = candidate_outputs[: args.max_outputs_per_sample]

			try:
				prompt_token_ids = torch.tensor([], dtype=torch.long)
				prompt_token_count: Optional[int] = None
				if "prompt_token_ids" in sample:
					pt_ids_raw = sample.get("prompt_token_ids")
					if isinstance(pt_ids_raw, (list, tuple)):
						prompt_token_ids = torch.tensor(list(pt_ids_raw), dtype=torch.long)
						prompt_token_count = len(prompt_token_ids)
				if "prompt_token_count" in sample:
					try:
						prompt_token_count = int(sample.get("prompt_token_count", 0))
					except (TypeError, ValueError):
						LOGGER.warning("Invalid prompt_token_count for sample %s; falling back to tokenization", sample_id)
				if prompt_token_count is None:
					prompt_token_count, prompt_token_ids = determine_prompt_length(
						tokenizer,
						prompt,
						system_prompt=system_prompt,
					)
			except Exception as exc:  # pylint: disable=broad-except
				LOGGER.exception("Failed to tokenize prompt for sample %s: %s", sample_id, exc)
				continue

			feature_map: Dict[int, Dict[str, object]] = {}
			for output_idx, candidate in enumerate(candidate_outputs):
				if not isinstance(candidate, str):
					LOGGER.warning(
						"Skipping non-string candidate at index %s for sample %s", output_idx, sample_id
					)
					continue
				candidate_code_range: Optional[Tuple[int, int]] = None
				raw_code_range = None
				if len(code_token_idx) > output_idx:
					raw_code_range = code_token_idx[output_idx]
					if isinstance(raw_code_range, (list, tuple)) and len(raw_code_range) >= 2:
						try:
							candidate_code_range = (
								int(raw_code_range[0]),
								int(raw_code_range[1]),
							)
						except (TypeError, ValueError):
							LOGGER.warning(
								"Invalid code token indices %s for sample %s candidate %s; skipping",
								raw_code_range,
								sample_id,
								output_idx,
							)
							candidate_code_range = None
					else:
						raw_code_range = None

				pretok_ids = None
				tok_field = sample.get("token_ids")
				if isinstance(tok_field, (list, tuple)) and len(tok_field) > output_idx:
					candidate_tok = tok_field[output_idx]
					if isinstance(candidate_tok, (list, tuple)):
						pretok_ids = list(candidate_tok)

				candidate_features = extract_for_candidate(
					tokenizer=tokenizer,
					model=model,
					prompt=prompt,
					candidate=candidate,
					prompt_token_count=prompt_token_count,
					targets=targets,
					code_token_range=candidate_code_range,
					system_prompt=system_prompt,
					reasoning_traces_enabled=reasoning_traces_enabled,
					reasoning_trace_delimiter=reasoning_trace_delimiter,
					include_token_entropy=args.include_token_entropy,
					token_entropy_top_k=args.token_entropy_top_k,
					include_logtoku=args.include_logtoku,
					logtoku_top_l=args.logtoku_top_l,
					pretok_ids=pretok_ids,
				)
		
				if len(is_correct) > output_idx:
					candidate_features["is_correct"] = is_correct[output_idx]
				if raw_code_range is not None:
					candidate_features["code_token_idx"] = raw_code_range

				feature_map[output_idx] = candidate_features

			payload = {
				"sample_id": sample_id,
				"language": language,
				"difficulty": difficulty,
				"prompt_token_ids": prompt_token_ids.detach().cpu(),
				"prompt_token_count": prompt_token_count,
				"hidden_state_layers": targets.hidden_state_layers,
				"attention_layers": targets.attention_layers,
				"features": feature_map,
			}

			save_path = save_sample_features(output_dir, split_name, sample_id, targets, payload)
			LOGGER.info("Saved features for sample %s to %s", sample_id, save_path)


if __name__ == "__main__":
	main()