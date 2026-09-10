from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Tuple

import matplotlib.pyplot as plt
from transformers import AutoTokenizer, PreTrainedTokenizerBase

POSITION_BUCKETS = 20

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Inspect token-level labels in augmented LiveCodeBench JSONL",
	)
	parser.add_argument(
		"--path",
		type=Path,
		required=True,
		help="Path to augmented livecodebench_<model>_augmented.jsonl",
	)
	parser.add_argument(
		"--save-prefix",
		default="token_labels",
		help="Prefix for saved plot filenames",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		help="Where to save plots (defaults to JSONL directory)",
	)
	parser.add_argument(
		"--tokenizer",
		default=None,
		help="Tokenizer name or path for whitespace/comment estimation",
	)
	parser.add_argument(
		"--token-labels-key",
		default="token_labels",
		help="Record key containing token-level labels",
	)
	parser.add_argument(
		"--line-error-map-key",
		default="line_error_map",
		help="Optional fallback key containing per-line error maps",
	)
	parser.add_argument(
		"--markdown-summary",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Write markdown summary in output dir (enabled by default; use --no-markdown-summary to disable)",
	)
	parser.add_argument(
		"--markdown-filename",
		default=None,
		help="Optional markdown filename (defaults to <save-prefix>_summary.md)",
	)
	return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
	with path.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def build_force_true_mask(
	label_tokenizer: PreTrainedTokenizerBase,
	code_response_token_ids: list[int],
) -> list[bool]:
	if not code_response_token_ids:
		return []
	code_text = label_tokenizer.decode(code_response_token_ids, skip_special_tokens=False)
	if not code_text:
		return [False for _ in code_response_token_ids]

	comment_spans: list[tuple[int, int]] = []
	for match in re.finditer(r"#.*", code_text):
		comment_spans.append((match.start(), match.end()))

	encoded = label_tokenizer(
		code_text,
		add_special_tokens=False,
		return_offsets_mapping=True,
	)
	offsets = encoded.get("offset_mapping") or []
	if len(offsets) != len(code_response_token_ids):
		return [False for _ in code_response_token_ids]

	force_true = []
	for start, end in offsets:
		if start is None or end is None or start >= end:
			force_true.append(False)
			continue
		substring = code_text[start:end]
		is_whitespace = bool(substring) and all(ch.isspace() for ch in substring)
		is_comment = any(start < span_end and end > span_start for span_start, span_end in comment_spans)
		force_true.append(is_whitespace or is_comment)
	return force_true


def extract_label_counts(
	*,
	token_labels: object,
	line_error_maps: object,
	label_tokenizer: PreTrainedTokenizerBase | None,
) -> Tuple[int, int, int, bool]:
	true_count = 0
	false_count = 0
	ws_comment_count = 0
	used_line_error_map = False

	if not isinstance(token_labels, list) or not token_labels:
		token_labels = []
	for candidate in token_labels:
		if not candidate:
			continue

		if isinstance(candidate, dict):
			candidate = candidate.get("tokens") or candidate.get("token_labels") or []
			if not isinstance(candidate, list):
				candidate = []

		if not isinstance(candidate, list) or not candidate:
			continue

		if isinstance(candidate[0], dict):
			for item in candidate:
				if not isinstance(item, dict):
					continue
				label = bool(item.get("label", False))
				if label:
					true_count += 1
				else:
					false_count += 1
				if item.get("ws_comment"):
					ws_comment_count += 1
			continue

		token_ids: list[int] = []
		for item in candidate:
			if not isinstance(item, (list, tuple)) or len(item) < 3:
				continue
			token_ids.append(int(item[0]))
			label = bool(item[2])
			if label:
				true_count += 1
			else:
				false_count += 1
		if label_tokenizer is not None and token_ids:
			force_true_mask = build_force_true_mask(label_tokenizer, token_ids)
			if len(force_true_mask) == len(token_ids):
				ws_comment_count += sum(1 for flag in force_true_mask if flag)

	if true_count + false_count == 0 and isinstance(line_error_maps, list) and line_error_maps:
		for candidate_map in line_error_maps:
			if not isinstance(candidate_map, dict):
				continue
			for line_is_clean in candidate_map.values():
				if bool(line_is_clean):
					true_count += 1
				else:
					false_count += 1
		used_line_error_map = true_count + false_count > 0

	return true_count, false_count, ws_comment_count, used_line_error_map


def _extract_line_id(item: dict) -> Hashable | None:
	for key in ("line", "line_idx", "line_id", "line_number"):
		value = item.get(key)
		if value is None:
			continue
		if isinstance(value, (int, str)):
			return value
		try:
			return int(value)
		except (TypeError, ValueError):
			continue
	return None


def iter_candidate_annotations(token_labels: object) -> Iterable[list[dict[str, object]]]:
	if not isinstance(token_labels, list):
		return
	for candidate in token_labels:
		if not candidate:
			continue
		if isinstance(candidate, dict):
			candidate = candidate.get("tokens") or candidate.get("token_labels") or []
			if not isinstance(candidate, list):
				candidate = []
		if not isinstance(candidate, list) or not candidate:
			continue

		normalized: list[dict[str, object]] = []
		if isinstance(candidate[0], dict):
			for item in candidate:
				if not isinstance(item, dict):
					continue
				normalized.append(
					{
						"label": bool(item.get("label", False)),
						"ws_comment": bool(item.get("ws_comment", False)),
						"line_id": _extract_line_id(item),
					}
				)
		elif isinstance(candidate[0], (list, tuple)):
			for item in candidate:
				if not isinstance(item, (list, tuple)) or len(item) < 3:
					continue
				line_id = None
				if len(item) >= 2:
					candidate_line = item[1]
					if isinstance(candidate_line, (int, str)):
						line_id = candidate_line
				normalized.append(
					{
						"label": bool(item[2]),
						"ws_comment": False,
						"line_id": line_id,
					}
				)

		if normalized:
			yield normalized


def percentile(values: list[float], p: float) -> float:
	if not values:
		return 0.0
	sorted_values = sorted(values)
	index = int(round((p / 100.0) * (len(sorted_values) - 1)))
	index = max(0, min(index, len(sorted_values) - 1))
	return sorted_values[index]


def pearson_correlation(x_vals: list[float], y_vals: list[float]) -> float | None:
	if len(x_vals) != len(y_vals) or len(x_vals) < 2:
		return None
	mean_x = sum(x_vals) / len(x_vals)
	mean_y = sum(y_vals) / len(y_vals)

	var_x = 0.0
	var_y = 0.0
	cov = 0.0
	for x_val, y_val in zip(x_vals, y_vals):
		dx = x_val - mean_x
		dy = y_val - mean_y
		var_x += dx * dx
		var_y += dy * dy
		cov += dx * dy
	if var_x <= 0.0 or var_y <= 0.0:
		return None
	return cov / math.sqrt(var_x * var_y)


def plot_label_distributions(
	*,
	percent_by_difficulty: Dict[str, Tuple[float, float, float]],
	aggregated_by_difficulty: Dict[str, Tuple[int, int, int]],
	save_path_percent: Path,
	save_path_agg: Path,
) -> None:
	difficulties = sorted(percent_by_difficulty.keys())
	if not difficulties:
		return
	labels = difficulties

	true_other_pcts = [percent_by_difficulty[d][0] for d in labels]
	ws_comment_pcts = [percent_by_difficulty[d][1] for d in labels]
	false_pcts = [percent_by_difficulty[d][2] for d in labels]

	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	ax.bar(labels, false_pcts, label="False", color="tab:red")
	bottom = false_pcts
	ax.bar(labels, true_other_pcts, bottom=bottom, label="True (non-ws/comment)", color="tab:green")
	bottom = [b + t for b, t in zip(bottom, true_other_pcts)]
	ax.bar(labels, ws_comment_pcts, bottom=bottom, label="Whitespace/comment (estimate)", color="tab:gray")
	ax.set_ylabel("Percent of tokens")
	ax.set_title("Average token-label distribution by difficulty (percent)")
	ax.legend()
	fig.tight_layout()
	save_path_percent.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(save_path_percent))
	plt.close(fig)

	true_other_counts = [aggregated_by_difficulty[d][0] for d in labels]
	ws_comment_counts = [aggregated_by_difficulty[d][1] for d in labels]
	false_counts = [aggregated_by_difficulty[d][2] for d in labels]

	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	ax.bar(labels, false_counts, label="False", color="tab:red")
	bottom = false_counts
	ax.bar(labels, true_other_counts, bottom=bottom, label="True (non-ws/comment)", color="tab:green")
	bottom = [b + t for b, t in zip(bottom, true_other_counts)]
	ax.bar(labels, ws_comment_counts, bottom=bottom, label="Whitespace/comment (estimate)", color="tab:gray")
	ax.set_ylabel("Token count")
	ax.set_title("Token-label distribution by difficulty (aggregated counts)")
	ax.legend()
	fig.tight_layout()
	save_path_agg.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(save_path_agg))
	plt.close(fig)


def plot_candidate_false_boxplot(
	*,
	false_pcts_by_difficulty: Dict[str, list[float]],
	save_path: Path,
) -> None:
	labels = sorted([key for key, values in false_pcts_by_difficulty.items() if values])
	if not labels:
		return
	data = [false_pcts_by_difficulty[label] for label in labels]

	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	ax.boxplot(data, showfliers=False)
	ax.set_xticks(range(1, len(labels) + 1), labels)
	ax.set_ylabel("Candidate false-label percent")
	ax.set_title("Candidate-level false-label distribution by difficulty")
	ax.set_ylim(0.0, 100.0)
	fig.tight_layout()
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(save_path))
	plt.close(fig)


def plot_length_vs_false_hexbin(
	*,
	candidate_lengths: list[int],
	candidate_false_pcts: list[float],
	save_path: Path,
) -> None:
	if not candidate_lengths or not candidate_false_pcts or len(candidate_lengths) != len(candidate_false_pcts):
		return

	fig, ax = plt.subplots(1, 1, figsize=(10, 6))
	hex_plot = ax.hexbin(candidate_lengths, candidate_false_pcts, gridsize=35, mincnt=1, cmap="viridis")
	ax.set_xlabel("Candidate token count")
	ax.set_ylabel("Candidate false-label percent")
	ax.set_title("Length vs. false-label rate (all candidates)")
	cb = fig.colorbar(hex_plot, ax=ax)
	cb.set_label("Candidates per bin")
	fig.tight_layout()
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(save_path))
	plt.close(fig)


def plot_position_error_profile(
	*,
	position_counts: Dict[str, list[list[int]]],
	save_path: Path,
) -> None:
	if not position_counts:
		return

	token_counts_by_diff: dict[str, int] = {}
	for diff, buckets in position_counts.items():
		token_counts_by_diff[diff] = sum(bucket[0] + bucket[1] for bucket in buckets)
	selected = sorted(
		[key for key, count in token_counts_by_diff.items() if count > 0],
		key=lambda key: token_counts_by_diff[key],
		reverse=True,
	)[:5]
	if not selected:
		return

	overall = [[0, 0] for _ in range(POSITION_BUCKETS)]
	for diff in selected:
		for bucket_idx in range(POSITION_BUCKETS):
			overall[bucket_idx][0] += position_counts[diff][bucket_idx][0]
			overall[bucket_idx][1] += position_counts[diff][bucket_idx][1]

	x = [100.0 * (idx + 0.5) / POSITION_BUCKETS for idx in range(POSITION_BUCKETS)]
	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	for diff in selected:
		y = []
		for true_count, false_count in position_counts[diff]:
			total = true_count + false_count
			y.append((100.0 * false_count / total) if total > 0 else 0.0)
		ax.plot(x, y, label=diff)

	y_overall = []
	for true_count, false_count in overall:
		total = true_count + false_count
		y_overall.append((100.0 * false_count / total) if total > 0 else 0.0)
	ax.plot(x, y_overall, linestyle="--", linewidth=2, color="black", label="overall(top5)")

	ax.set_xlabel("Relative token position in candidate (%)")
	ax.set_ylabel("False-label rate (%)")
	ax.set_title("Position-wise false-label profile")
	ax.set_ylim(0.0, 100.0)
	ax.legend()
	fig.tight_layout()
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(save_path))
	plt.close(fig)


def write_markdown_summary(
	*,
	save_path: Path,
	input_path: Path,
	overall_token_distribution: dict[str, float | int],
	overall_line_distribution: dict[str, float | int],
	overall_length_stats: dict[str, float | int],
	per_difficulty_payload: dict[str, dict[str, Any]],
	fallback_rows: int,
	line_error_map_key: str,
) -> None:
	lines: list[str] = []
	lines.append("# Token Label Summary")
	lines.append("")
	lines.append(f"- Source: `{input_path}`")
	lines.append("")
	lines.append("## Overall")
	lines.append("")
	lines.append("### Token-level label distribution (excluding ws_comment tokens)")
	lines.append("")
	lines.append(f"- True: {overall_token_distribution['true']} ({overall_token_distribution['true_pct']}%)")
	lines.append(f"- False: {overall_token_distribution['false']} ({overall_token_distribution['false_pct']}%)")
	lines.append(f"- Total: {overall_token_distribution['total']}")
	lines.append("")
	lines.append("### Line-level label distribution (excluding ws_comment-only lines)")
	lines.append("")
	lines.append(f"- True: {overall_line_distribution['true']} ({overall_line_distribution['true_pct']}%)")
	lines.append(f"- False: {overall_line_distribution['false']} ({overall_line_distribution['false_pct']}%)")
	lines.append(f"- Total: {overall_line_distribution['total']}")
	lines.append("")
	lines.append("### Average sample lengths")
	lines.append("")
	lines.append(f"- Tokens mean (excl. ws_comment): {overall_length_stats['tokens_mean']}")
	lines.append(f"- Lines mean (excl. ws_comment-only): {overall_length_stats['lines_mean']}")
	lines.append(f"- N samples (token lengths): {overall_length_stats['n_samples_tokens']}")
	lines.append(f"- N samples (line lengths): {overall_length_stats['n_samples_lines']}")

	lines.append("")
	lines.append("## Per difficulty")
	for difficulty in sorted(per_difficulty_payload.keys()):
		payload = per_difficulty_payload[difficulty]
		token_stats = payload["token_level_excl_ws_comment"]
		line_stats = payload["line_level_excl_ws_comment_only_lines"]
		lines.append("")
		lines.append(f"### {difficulty}")
		lines.append("")
		lines.append(
			f"- Token-level (excl. ws_comment): true={token_stats['true']} ({token_stats['true_pct']}%), "
			f"false={token_stats['false']} ({token_stats['false_pct']}%), total={token_stats['total']}"
		)
		lines.append(
			f"- Line-level (excl. ws_comment-only): true={line_stats['true']} ({line_stats['true_pct']}%), "
			f"false={line_stats['false']} ({line_stats['false_pct']}%), total={line_stats['total']}"
		)
		lines.append(f"- Avg sample len tokens: {payload['avg_sample_len_tokens']} (n={payload['n_samples_tokens']})")
		lines.append(f"- Avg sample len lines: {payload['avg_sample_len_lines']} (n={payload['n_samples_lines']})")

	if fallback_rows > 0:
		lines.append("")
		lines.append("## Notes")
		lines.append("")
		lines.append(
			f"- Used `{line_error_map_key}` fallback for {fallback_rows} rows (counts are line-level for those rows)."
		)

	save_path.parent.mkdir(parents=True, exist_ok=True)
	save_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
	args = parse_args()
	path: Path = args.path
	if not path.exists():
		raise FileNotFoundError(path)
	output_dir = args.output_dir or path.parent

	label_tokenizer: PreTrainedTokenizerBase | None = None
	if args.tokenizer:
		label_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
	else:
		print("No tokenizer provided; whitespace/comment estimate disabled for legacy labels.")

	difficulty_counts: Counter[str] = Counter()
	agg_counts: Dict[str, Tuple[int, int, int]] = defaultdict(lambda: (0, 0, 0))
	sum_pct: Dict[str, Tuple[float, float, float]] = defaultdict(lambda: (0.0, 0.0, 0.0))
	sample_counts: Counter[str] = Counter()
	fallback_rows = 0
	candidate_false_pcts_by_difficulty: dict[str, list[float]] = defaultdict(list)
	candidate_lengths_by_difficulty: dict[str, list[int]] = defaultdict(list)
	position_counts_by_difficulty: dict[str, list[list[int]]] = defaultdict(
		lambda: [[0, 0] for _ in range(POSITION_BUCKETS)]
	)
	row_false_rate_std_by_difficulty: dict[str, list[float]] = defaultdict(list)
	token_metric_candidates = 0
	filtered_token_counts_by_difficulty: dict[str, list[int]] = defaultdict(lambda: [0, 0])
	filtered_line_counts_by_difficulty: dict[str, list[int]] = defaultdict(lambda: [0, 0])
	sample_token_lengths_by_difficulty: dict[str, list[int]] = defaultdict(list)
	sample_line_lengths_by_difficulty: dict[str, list[int]] = defaultdict(list)

	for rec in iter_jsonl(path):
		difficulty = str(rec.get("difficulty", "unknown")).lower()
		difficulty_counts[difficulty] += 1
		true_count, false_count, ws_comment_count, used_line_error_map = extract_label_counts(
			token_labels=rec.get(args.token_labels_key),
			line_error_maps=rec.get(args.line_error_map_key),
			label_tokenizer=label_tokenizer,
		)
		if used_line_error_map:
			fallback_rows += 1

		candidate_false_rates_for_row: list[float] = []
		for candidate_tokens in iter_candidate_annotations(rec.get(args.token_labels_key)):
			candidate_len = len(candidate_tokens)
			if candidate_len == 0:
				continue

			non_ws_tokens = [item for item in candidate_tokens if not bool(item["ws_comment"])]
			if non_ws_tokens:
				non_ws_true = sum(1 for item in non_ws_tokens if bool(item["label"]))
				non_ws_false = len(non_ws_tokens) - non_ws_true
				filtered_token_counts_by_difficulty[difficulty][0] += non_ws_true
				filtered_token_counts_by_difficulty[difficulty][1] += non_ws_false
				sample_token_lengths_by_difficulty[difficulty].append(len(non_ws_tokens))

			lines_to_tokens: dict[Hashable, list[dict[str, object]]] = defaultdict(list)
			for item in candidate_tokens:
				line_id = item.get("line_id")
				if line_id is None:
					continue
				lines_to_tokens[line_id].append(item)
			if lines_to_tokens:
				effective_line_count = 0
				for tokens_in_line in lines_to_tokens.values():
					non_ws_line_tokens = [tok for tok in tokens_in_line if not bool(tok["ws_comment"])]
					if not non_ws_line_tokens:
						continue
					effective_line_count += 1
					line_is_true = all(bool(tok["label"]) for tok in non_ws_line_tokens)
					if line_is_true:
						filtered_line_counts_by_difficulty[difficulty][0] += 1
					else:
						filtered_line_counts_by_difficulty[difficulty][1] += 1
				if effective_line_count > 0:
					sample_line_lengths_by_difficulty[difficulty].append(effective_line_count)

			false_count_candidate = sum(1 for item in candidate_tokens if not item["label"])
			false_pct_candidate = 100.0 * false_count_candidate / candidate_len
			candidate_false_rates_for_row.append(false_pct_candidate)
			candidate_false_pcts_by_difficulty[difficulty].append(false_pct_candidate)
			candidate_lengths_by_difficulty[difficulty].append(candidate_len)
			token_metric_candidates += 1

			for idx, item in enumerate(candidate_tokens):
				bucket = min((idx * POSITION_BUCKETS) // candidate_len, POSITION_BUCKETS - 1)
				if item["label"]:
					position_counts_by_difficulty[difficulty][bucket][0] += 1
				else:
					position_counts_by_difficulty[difficulty][bucket][1] += 1

		if len(candidate_false_rates_for_row) >= 2:
			row_false_rate_std_by_difficulty[difficulty].append(
				statistics.pstdev(candidate_false_rates_for_row)
			)

		if true_count + false_count == 0:
			continue
		sample_counts[difficulty] += 1
		total = true_count + false_count
		true_pct = true_count / total
		false_pct = false_count / total
		ws_comment_pct = ws_comment_count / total if total > 0 else 0.0
		prev_true_pct, prev_false_pct, prev_ws_pct = sum_pct[difficulty]
		sum_pct[difficulty] = (
			prev_true_pct + true_pct,
			prev_false_pct + false_pct,
			prev_ws_pct + ws_comment_pct,
		)
		prev_true_cnt, prev_false_cnt, prev_ws_cnt = agg_counts[difficulty]
		agg_counts[difficulty] = (
			prev_true_cnt + true_count,
			prev_false_cnt + false_count,
			prev_ws_cnt + ws_comment_count,
		)

	# Overall aggregation
	overall_true = sum(v[0] for v in agg_counts.values())
	overall_false = sum(v[1] for v in agg_counts.values())
	overall_ws = sum(v[2] for v in agg_counts.values())
	overall_samples = sum(sample_counts.values())
	overall_true_pct = 0.0
	overall_false_pct = 0.0
	overall_ws_pct = 0.0
	if overall_samples > 0:
		overall_true_pct = sum(v[0] for v in sum_pct.values()) / overall_samples
		overall_false_pct = sum(v[1] for v in sum_pct.values()) / overall_samples
		overall_ws_pct = sum(v[2] for v in sum_pct.values()) / overall_samples

	percent_by_difficulty: Dict[str, Tuple[float, float, float]] = {}
	aggregated_by_difficulty: Dict[str, Tuple[int, int, int]] = {}
	for diff in sorted(difficulty_counts.keys()):
		if sample_counts[diff] > 0:
			sum_true_pct, sum_false_pct, sum_ws_pct = sum_pct[diff]
			avg_true_pct = sum_true_pct / sample_counts[diff]
			avg_false_pct = sum_false_pct / sample_counts[diff]
			avg_ws_pct = min(sum_ws_pct / sample_counts[diff], avg_true_pct)
			avg_true_other_pct = max(avg_true_pct - avg_ws_pct, 0.0)
			percent_by_difficulty[diff] = (
				avg_true_other_pct * 100.0,
				avg_ws_pct * 100.0,
				avg_false_pct * 100.0,
			)
		else:
			percent_by_difficulty[diff] = (0.0, 0.0, 0.0)
		true_count, false_count, ws_count = agg_counts[diff]
		ws_count = min(ws_count, true_count)
		aggregated_by_difficulty[diff] = (true_count - ws_count, ws_count, false_count)

	overall_ws_pct = min(overall_ws_pct, overall_true_pct)
	overall_true_other_pct = max(overall_true_pct - overall_ws_pct, 0.0)
	percent_by_difficulty["overall"] = (
		overall_true_other_pct * 100.0,
		overall_ws_pct * 100.0,
		overall_false_pct * 100.0,
	)
	overall_ws = min(overall_ws, overall_true)
	aggregated_by_difficulty["overall"] = (overall_true - overall_ws, overall_ws, overall_false)

	print("Samples per difficulty (with token labels):", dict(sample_counts))
	print("Overall token counts:", {"true": overall_true, "false": overall_false, "ws_comment": overall_ws})

	def distribution_payload(true_count: int, false_count: int) -> dict[str, float | int]:
		total = true_count + false_count
		if total <= 0:
			return {
				"true": true_count,
				"false": false_count,
				"total": total,
				"true_pct": 0.0,
				"false_pct": 0.0,
			}
		return {
			"true": true_count,
			"false": false_count,
			"total": total,
			"true_pct": round(100.0 * true_count / total, 2),
			"false_pct": round(100.0 * false_count / total, 2),
		}

	def mean_or_zero(values: list[int]) -> float:
		if not values:
			return 0.0
		return round(sum(values) / len(values), 2)

	overall_filtered_token_true = sum(values[0] for values in filtered_token_counts_by_difficulty.values())
	overall_filtered_token_false = sum(values[1] for values in filtered_token_counts_by_difficulty.values())
	overall_filtered_line_true = sum(values[0] for values in filtered_line_counts_by_difficulty.values())
	overall_filtered_line_false = sum(values[1] for values in filtered_line_counts_by_difficulty.values())
	all_sample_token_lengths = [value for values in sample_token_lengths_by_difficulty.values() for value in values]
	all_sample_line_lengths = [value for values in sample_line_lengths_by_difficulty.values() for value in values]
	overall_token_distribution = distribution_payload(overall_filtered_token_true, overall_filtered_token_false)
	overall_line_distribution = distribution_payload(overall_filtered_line_true, overall_filtered_line_false)
	overall_length_stats = {
		"tokens_mean": mean_or_zero(all_sample_token_lengths),
		"lines_mean": mean_or_zero(all_sample_line_lengths),
		"n_samples_tokens": len(all_sample_token_lengths),
		"n_samples_lines": len(all_sample_line_lengths),
	}

	print(
		"Token-level label distribution (excluding ws_comment tokens):",
		overall_token_distribution,
	)
	print(
		"Line-level label distribution (excluding ws_comment-only lines):",
		overall_line_distribution,
	)
	print(
		"Average sample lengths (excluding ws_comment tokens/lines):",
		overall_length_stats,
	)
	per_difficulty_payload: dict[str, dict[str, Any]] = {}
	for difficulty in sorted(difficulty_counts.keys()):
		token_true, token_false = filtered_token_counts_by_difficulty[difficulty]
		line_true, line_false = filtered_line_counts_by_difficulty[difficulty]
		difficulty_payload = {
			"token_level_excl_ws_comment": distribution_payload(token_true, token_false),
			"line_level_excl_ws_comment_only_lines": distribution_payload(line_true, line_false),
			"avg_sample_len_tokens": mean_or_zero(sample_token_lengths_by_difficulty[difficulty]),
			"avg_sample_len_lines": mean_or_zero(sample_line_lengths_by_difficulty[difficulty]),
			"n_samples_tokens": len(sample_token_lengths_by_difficulty[difficulty]),
			"n_samples_lines": len(sample_line_lengths_by_difficulty[difficulty]),
		}
		per_difficulty_payload[difficulty] = difficulty_payload
		print(
			f"Difficulty distribution [{difficulty}]",
			difficulty_payload,
		)
	if fallback_rows > 0:
		print(
			f"Used {args.line_error_map_key} fallback for {fallback_rows} rows "
			"(counts are line-level for those rows)."
		)

	all_lengths = [value for values in candidate_lengths_by_difficulty.values() for value in values]
	all_false_pcts = [value for values in candidate_false_pcts_by_difficulty.values() for value in values]
	print("Token-level candidates used for advanced analytics:", token_metric_candidates)
	if all_false_pcts:
		print(
			"Overall candidate false%:",
			{
				"mean": round(sum(all_false_pcts) / len(all_false_pcts), 2),
				"median": round(statistics.median(all_false_pcts), 2),
				"p90": round(percentile(all_false_pcts, 90.0), 2),
			},
		)
	if all_lengths and all_false_pcts and len(all_lengths) == len(all_false_pcts):
		corr = pearson_correlation([float(value) for value in all_lengths], all_false_pcts)
		if corr is not None:
			print("Length vs false% Pearson r:", round(corr, 4))

	for difficulty in sorted(candidate_false_pcts_by_difficulty.keys()):
		values = candidate_false_pcts_by_difficulty[difficulty]
		lengths = candidate_lengths_by_difficulty[difficulty]
		if not values:
			continue
		stats_payload = {
			"n_candidates": len(values),
			"false_mean": round(sum(values) / len(values), 2),
			"false_median": round(statistics.median(values), 2),
			"false_p90": round(percentile(values, 90.0), 2),
			"len_mean": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
		}
		spread_values = row_false_rate_std_by_difficulty.get(difficulty, [])
		if spread_values:
			stats_payload["row_false_std_mean"] = round(sum(spread_values) / len(spread_values), 2)
		print(f"Difficulty summary [{difficulty}]", stats_payload)

	plot_percent_path = output_dir / f"{args.save_prefix}_label_distribution_percent.png"
	plot_agg_path = output_dir / f"{args.save_prefix}_label_distribution_aggregated.png"
	plot_false_box_path = output_dir / f"{args.save_prefix}_candidate_false_boxplot.png"
	plot_len_false_path = output_dir / f"{args.save_prefix}_length_vs_false_hexbin.png"
	plot_position_profile_path = output_dir / f"{args.save_prefix}_position_error_profile.png"
	plot_label_distributions(
		percent_by_difficulty=percent_by_difficulty,
		aggregated_by_difficulty=aggregated_by_difficulty,
		save_path_percent=plot_percent_path,
		save_path_agg=plot_agg_path,
	)
	plot_candidate_false_boxplot(
		false_pcts_by_difficulty=candidate_false_pcts_by_difficulty,
		save_path=plot_false_box_path,
	)
	plot_length_vs_false_hexbin(
		candidate_lengths=all_lengths,
		candidate_false_pcts=all_false_pcts,
		save_path=plot_len_false_path,
	)
	plot_position_error_profile(
		position_counts=position_counts_by_difficulty,
		save_path=plot_position_profile_path,
	)
	markdown_path: Path | None = None
	if args.markdown_summary:
		markdown_path_local = output_dir / (args.markdown_filename or f"{args.save_prefix}_summary.md")
		write_markdown_summary(
			save_path=markdown_path_local,
			input_path=path,
			overall_token_distribution=overall_token_distribution,
			overall_line_distribution=overall_line_distribution,
			overall_length_stats=overall_length_stats,
			per_difficulty_payload=per_difficulty_payload,
			fallback_rows=fallback_rows,
			line_error_map_key=args.line_error_map_key,
		)
		markdown_path = markdown_path_local
	print(f"Saved percent plot to {plot_percent_path}")
	print(f"Saved aggregated plot to {plot_agg_path}")
	print(f"Saved candidate false-rate boxplot to {plot_false_box_path}")
	print(f"Saved length-vs-false hexbin to {plot_len_false_path}")
	print(f"Saved position error profile to {plot_position_profile_path}")
	if markdown_path is not None:
		print(f"Saved markdown summary to {markdown_path}")


if __name__ == "__main__":
	main()	