"""
Split LiveCodeBench JSONL into train/validation/test files.

Default behavior:
- Uses ratios `train=0.6`, `val=0.2`, `test=0.2` with `seed=42`.
- Shuffles rows and writes `<base>_train.jsonl`, `<base>_val.jsonl`, `<base>_test.jsonl`.
- Writes to `--output-dir` or the input file parent directory by default.
- No stratification unless `--stratify` is enabled.

Important options:
- `--stratify`: preserves ratio of `any(is_correct)` labels across splits.
- `--train-ratio`, `--val-ratio`, `--test-ratio`: must sum to 1.0.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Split LiveCodeBench JSONL into train/val/test")
	parser.add_argument("--input", type=Path, required=True, help="Path to input JSONL produced by build_dataset.py")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory for split files; defaults to input parent",
	)
	parser.add_argument("--train-ratio", type=float, default=0.6, help="Train split fraction")
	parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split fraction")
	parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split fraction")
	parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
	parser.add_argument(
		"--stratify",
		action="store_true",
		help="Stratify on any(record['is_correct']) to preserve label ratios",
	)
	return parser.parse_args()


def _validate_ratios(train: float, val: float, test: float) -> None:
	total = train + val + test
	if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-6):
		raise ValueError(f"Ratios must sum to 1.0; got {total:.6f}")
	for name, value in ("train", train), ("val", val), ("test", test):
		if value < 0:
			raise ValueError(f"{name} ratio must be non-negative; got {value}")


def _load_records(path: Path) -> List[dict]:
	with path.open("r", encoding="utf-8") as f:
		return [json.loads(line) for line in f if line.strip()]


def _allocate_counts(bucket_size: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
	# Allocate per-bucket counts so they sum exactly to bucket_size using fractional rounding.
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


def split_records(records: List[dict], ratios: Tuple[float, float, float], stratify: bool, seed: int):
	rng = random.Random(seed)

	if stratify:
		buckets = {True: [], False: []}
		for idx, rec in enumerate(records):
			label = bool(any(rec.get("is_correct", [])))
			buckets[label].append(idx)
		train_idx: list[int] = []
		val_idx: list[int] = []
		test_idx: list[int] = []
		for lbl_bucket in buckets.values():
			rng.shuffle(lbl_bucket)
			t_count, v_count, s_count = _allocate_counts(len(lbl_bucket), ratios)
			train_idx.extend(lbl_bucket[:t_count])
			val_idx.extend(lbl_bucket[t_count : t_count + v_count])
			test_idx.extend(lbl_bucket[t_count + v_count : t_count + v_count + s_count])
		# final shuffle to avoid label-order artifacts
		rng.shuffle(train_idx)
		rng.shuffle(val_idx)
		rng.shuffle(test_idx)
	else:
		all_idx = list(range(len(records)))
		rng.shuffle(all_idx)
		t_count, v_count, s_count = _allocate_counts(len(all_idx), ratios)
		train_idx = all_idx[:t_count]
		val_idx = all_idx[t_count : t_count + v_count]
		test_idx = all_idx[t_count + v_count : t_count + v_count + s_count]

	def gather(idxs: List[int]) -> List[dict]:
		return [records[i] for i in idxs]

	return gather(train_idx), gather(val_idx), gather(test_idx)


def main() -> None:
	args = parse_args()
	_validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

	input_path = args.input
	output_dir = args.output_dir or input_path.parent
	base = input_path.stem

	records = _load_records(input_path)
	train_ratio = args.train_ratio
	val_ratio = args.val_ratio
	test_ratio = args.test_ratio

	train_set, val_set, test_set = split_records(
		records,
		(train_ratio, val_ratio, test_ratio),
		stratify=args.stratify,
		seed=args.seed,
	)

	_write_records(output_dir / f"{base}_train.jsonl", train_set)
	_write_records(output_dir / f"{base}_val.jsonl", val_set)
	_write_records(output_dir / f"{base}_test.jsonl", test_set)

	total = len(records)
	print(f"Wrote {len(train_set)} train, {len(val_set)} val, {len(test_set)} test (total {total}) to {output_dir}")


if __name__ == "__main__":
	main()