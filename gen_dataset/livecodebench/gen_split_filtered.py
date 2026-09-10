"""
Create grouped, response-stratified splits for filtered augmented LiveCodeBench data.

Default behavior:
- Uses ratios `train=0.6`, `val=0.2`, `test=0.2` with `seed=42`.
- Performs strict row-level grouping (a row stays in exactly one split).
- Optimizes assignment to preserve candidate-level correctness distribution (`is_correct`)
	and candidate counts across splits.
- Writes `<base>_train.jsonl`, `<base>_val.jsonl`, `<base>_test.jsonl`.

Important options:
- `--output-dir`: defaults to input parent directory.
- `--train-ratio`, `--val-ratio`, `--test-ratio`: must sum to 1.0.
- `--seed`: deterministic assignment/shuffling.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class RowStats:
	idx: int
	total_candidates: int
	correct_candidates: int


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Split filtered augmented LiveCodeBench JSONL into train/val/test with strict "
			"problem-level grouping and response-level (candidate) stratification on is_correct."
		)
	)
	parser.add_argument("--input", type=Path, required=True, help="Path to filtered augmented input JSONL")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory for split files; defaults to input parent",
	)
	parser.add_argument("--train-ratio", type=float, default=0.6, help="Train split fraction")
	parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split fraction")
	parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split fraction")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic assignment")
	return parser.parse_args()


def _validate_ratios(train: float, val: float, test: float) -> None:
	total = train + val + test
	if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-6):
		raise ValueError(f"Ratios must sum to 1.0; got {total:.6f}")
	for name, value in (("train", train), ("val", val), ("test", test)):
		if value < 0:
			raise ValueError(f"{name} ratio must be non-negative; got {value}")


def _load_records(path: Path) -> List[dict]:
	with path.open("r", encoding="utf-8") as f:
		return [json.loads(line) for line in f if line.strip()]


def _allocate_counts(bucket_size: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
	exact = [r * bucket_size for r in ratios]
	bases = [int(x) for x in exact]
	remainder = bucket_size - sum(bases)
	frac_idx = sorted(range(len(ratios)), key=lambda i: exact[i] - bases[i], reverse=True)
	for i in frac_idx[:remainder]:
		bases[i] += 1
	return bases[0], bases[1], bases[2]


def _write_records(path: Path, records: Iterable[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for rec in records:
			f.write(json.dumps(rec) + "\n")


def _extract_row_stats(records: Sequence[dict]) -> list[RowStats]:
	rows: list[RowStats] = []
	for idx, rec in enumerate(records):
		flags = rec.get("is_correct", [])
		if not isinstance(flags, list):
			flags = []
		total = len(flags)
		correct = sum(1 for value in flags if bool(value))
		rows.append(RowStats(idx=idx, total_candidates=total, correct_candidates=correct))
	return rows


def _assignment_cost(
	*,
	split_idx: int,
	row: RowStats,
	current_rows: list[int],
	current_totals: list[int],
	current_correct: list[int],
	target_rows: tuple[int, int, int],
	target_totals: tuple[int, int, int],
	target_correct: tuple[int, int, int],
) -> float:
	rows_after = current_rows[:]
	totals_after = current_totals[:]
	correct_after = current_correct[:]

	rows_after[split_idx] += 1
	totals_after[split_idx] += row.total_candidates
	correct_after[split_idx] += row.correct_candidates

	total_err = sum(abs(totals_after[i] - target_totals[i]) for i in range(3))
	correct_err = sum(abs(correct_after[i] - target_correct[i]) for i in range(3))
	rows_err = sum(abs(rows_after[i] - target_rows[i]) for i in range(3))

	return (1.0 * float(total_err)) + (2.0 * float(correct_err)) + (0.2 * float(rows_err))


def split_records_grouped_stratified(
	records: List[dict],
	ratios: Tuple[float, float, float],
	seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
	rng = random.Random(seed)
	row_stats = _extract_row_stats(records)
	rng.shuffle(row_stats)

	global_correct_rate = 0.0
	global_total = sum(row.total_candidates for row in row_stats)
	if global_total > 0:
		global_correct_rate = sum(row.correct_candidates for row in row_stats) / float(global_total)

	row_stats.sort(
		key=lambda row: (
			row.total_candidates,
			abs((row.correct_candidates / row.total_candidates) - global_correct_rate)
			if row.total_candidates > 0
			else 0.0,
		),
		reverse=True,
	)

	target_rows = _allocate_counts(len(row_stats), ratios)
	target_totals = _allocate_counts(global_total, ratios)
	target_correct = _allocate_counts(sum(row.correct_candidates for row in row_stats), ratios)

	assignment: list[list[int]] = [[], [], []]
	current_rows = [0, 0, 0]
	current_totals = [0, 0, 0]
	current_correct = [0, 0, 0]

	for row in row_stats:
		best_split = 0
		best_cost = None
		for split_idx in range(3):
			cost = _assignment_cost(
				split_idx=split_idx,
				row=row,
				current_rows=current_rows,
				current_totals=current_totals,
				current_correct=current_correct,
				target_rows=target_rows,
				target_totals=target_totals,
				target_correct=target_correct,
			)
			if best_cost is None or cost < best_cost:
				best_cost = cost
				best_split = split_idx

		assignment[best_split].append(row.idx)
		current_rows[best_split] += 1
		current_totals[best_split] += row.total_candidates
		current_correct[best_split] += row.correct_candidates

	def gather(idxs: list[int]) -> list[dict]:
		return [records[i] for i in idxs]

	return gather(assignment[0]), gather(assignment[1]), gather(assignment[2])


def _compute_split_candidate_stats(records: Sequence[dict]) -> tuple[int, int, int]:
	total = 0
	correct = 0
	for rec in records:
		flags = rec.get("is_correct", [])
		if not isinstance(flags, list):
			continue
		total += len(flags)
		correct += sum(1 for value in flags if bool(value))
	incorrect = total - correct
	return total, correct, incorrect


def _print_summary(
	*,
	train_set: Sequence[dict],
	val_set: Sequence[dict],
	test_set: Sequence[dict],
	output_dir: Path,
) -> None:
	splits = {
		"train": train_set,
		"val": val_set,
		"test": test_set,
	}
	global_total, global_correct, _ = _compute_split_candidate_stats(
		list(train_set) + list(val_set) + list(test_set)
	)
	global_rate = (global_correct / global_total) if global_total else 0.0

	print(f"Wrote grouped stratified splits to {output_dir}")
	for name in SPLIT_NAMES:
		records = splits[name]
		cand_total, cand_correct, cand_incorrect = _compute_split_candidate_stats(records)
		rate = (cand_correct / cand_total) if cand_total else 0.0
		delta = abs(rate - global_rate)
		print(
			f"{name}: rows={len(records)} candidates={cand_total} "
			f"correct={cand_correct} incorrect={cand_incorrect} "
			f"correct_rate={rate:.4f} (|Δglobal|={delta:.4f})"
		)


def main() -> None:
	args = parse_args()
	_validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

	input_path = args.input
	output_dir = args.output_dir or input_path.parent
	base = input_path.stem

	records = _load_records(input_path)
	train_set, val_set, test_set = split_records_grouped_stratified(
		records,
		(args.train_ratio, args.val_ratio, args.test_ratio),
		seed=args.seed,
	)

	_write_records(output_dir / f"{base}_train.jsonl", train_set)
	_write_records(output_dir / f"{base}_val.jsonl", val_set)
	_write_records(output_dir / f"{base}_test.jsonl", test_set)

	_print_summary(
		train_set=train_set,
		val_set=val_set,
		test_set=test_set,
		output_dir=output_dir,
	)


if __name__ == "__main__":
	main()