"""
Re-split BigCodeBench feature files into domain-holdout train/val/test partitions.

Default behavior:
- Scans feature files under `--data-dir` across splits
	`train`, `validation`, `test`.
- Loads BigCodeBench metadata from `bigcode/bigcodebench` (`default`, `v0.1.4`).
- Maps each task's libraries to domains using `bcb_domains.json`.
- Sends all files matching `--target-domain` to the new `test` split.
- Splits remaining files into `train`/`validation` with an 80/20 stratified split
	over response correctness labels (derived from `is_correct`).
- If `--output-dir` is omitted, writes to `<data_dir>_domain_<target-domain>`.

Important options:
- Dataset mapping source: `--dataset-name`, `--dataset-config`, `--dataset-revision`.
- Scan scope and reproducibility: `--splits`, `--seed`, `--log-level`.
- `--target-domain` is required and defines the holdout test domain.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import random
import shutil
from collections.abc import Iterable as IterableABC, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from datasets import load_dataset


LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "bigcode/bigcodebench"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_DATASET_REVISION = "v0.1.4"


@dataclass
class FileRecord:
	path: Path
	sample_id: str
	label: bool
	domains: Set[str]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Reassemble BigCodeBench feature files by domain")
	parser.add_argument("--data-dir", type=Path, required=True, help="Directory with train/validation/test .pt files")
	parser.add_argument("--target-domain", type=str, required=True, help="Domain to hold out for test split")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Output directory for rearranged splits (defaults to data-dir + domain suffix)",
	)
	parser.add_argument(
		"--dataset-name",
		default=DEFAULT_DATASET_NAME,
		help="HuggingFace dataset name for BigCodeBench",
	)
	parser.add_argument(
		"--dataset-config",
		default=DEFAULT_DATASET_CONFIG,
		help="Dataset config name",
	)
	parser.add_argument(
		"--dataset-revision",
		default=DEFAULT_DATASET_REVISION,
		help="Dataset revision/tag",
	)
	parser.add_argument(
		"--splits",
		nargs="*",
		default=("train", "validation", "test"),
		help="Input split directories to scan",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
	parser.add_argument("--log-level", default="INFO", help="Logging level (e.g., INFO, DEBUG)")
	return parser.parse_args()


def setup_logging(level: str) -> None:
	logging.basicConfig(
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		level=getattr(logging, level.upper(), logging.INFO),
	)


def _allocate_counts(bucket_size: int, ratios: Tuple[float, float]) -> Tuple[int, int]:
	# Allocate per-bucket counts so they sum exactly to bucket_size using fractional rounding.
	exact = [r * bucket_size for r in ratios]
	bases = [int(x) for x in exact]
	remainder = bucket_size - sum(bases)
	frac_idx = sorted(range(len(ratios)), key=lambda i: exact[i] - bases[i], reverse=True)
	for i in frac_idx[:remainder]:
		bases[i] += 1
	return bases[0], bases[1]


def _validate_ratios(train: float, val: float) -> None:
	total = train + val
	if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-6):
		raise ValueError(f"Ratios must sum to 1.0; got {total:.6f}")
	for name, value in ("train", train), ("val", val):
		if value < 0:
			raise ValueError(f"{name} ratio must be non-negative; got {value}")


def split_records(records: List[FileRecord], ratios: Tuple[float, float], seed: int) -> Tuple[List[FileRecord], List[FileRecord]]:
	rng = random.Random(seed)
	buckets = {True: [], False: []}
	for idx, rec in enumerate(records):
		buckets[rec.label].append(idx)

	train_idx: List[int] = []
	val_idx: List[int] = []
	for lbl_bucket in buckets.values():
		rng.shuffle(lbl_bucket)
		t_count, v_count = _allocate_counts(len(lbl_bucket), ratios)
		train_idx.extend(lbl_bucket[:t_count])
		val_idx.extend(lbl_bucket[t_count : t_count + v_count])

	# final shuffle to avoid label-order artifacts
	rng.shuffle(train_idx)
	rng.shuffle(val_idx)

	def gather(idxs: List[int]) -> List[FileRecord]:
		return [records[i] for i in idxs]

	return gather(train_idx), gather(val_idx)


def load_domain_map(path: Path) -> Dict[str, str]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError(f"Expected mapping in {path}; got {type(data)}")
	return {str(k): str(v) for k, v in data.items()}


def normalize_libs(raw: object) -> List[str]:
	if isinstance(raw, (list, tuple)):
		return [str(item) for item in raw if isinstance(item, str)]
	if isinstance(raw, str):
		text = raw.strip()
		if not text:
			return []
		try:
			parsed = ast.literal_eval(text)
			if isinstance(parsed, (list, tuple)):
				return [str(item) for item in parsed if isinstance(item, str)]
		except (SyntaxError, ValueError):
			pass
		if "," in text:
			parts = [p.strip().strip("'\"") for p in text.strip("[]").split(",")]
			return [p for p in parts if p]
		return [text.strip("'\"")]
	return []



def build_task_domain_index(
	dataset_name: str,
	dataset_config: str,
	dataset_revision: Optional[str],
	lib_to_domain: Dict[str, str],
) -> Tuple[Dict[str, Set[str]], Set[str]]:
	if dataset_revision is None:
		raise ValueError("dataset_revision must be provided and used as the dataset split (e.g. v0.1.4)")
	if dataset_config and dataset_config != "default":
		dataset_name = f"{dataset_name}-{dataset_config}"
	LOGGER.info("Loading dataset %s with split %s", dataset_name, dataset_revision)
	dataset = load_dataset(dataset_name, split=dataset_revision)
	lib_to_domain_lower = {k.lower(): v for k, v in lib_to_domain.items()}
	unknown_libs: Set[str] = set()
	index: Dict[str, Set[str]] = {}

	split_len = len(dataset) if isinstance(dataset, Sized) else "unknown"
	LOGGER.info("Indexing %s rows", split_len)
	split_iter = dataset if isinstance(dataset, IterableABC) else []
	for sample in split_iter:
			if not isinstance(sample, dict):
				continue
			task_id = sample.get("task_id") or sample.get("id")
			if task_id is None:
				continue
			libs = normalize_libs(sample.get("libs"))
			domains: Set[str] = set()
			for lib in libs:
				domain = lib_to_domain.get(lib)
				if domain is None:
					domain = lib_to_domain_lower.get(lib.lower())
				if domain is None:
					unknown_libs.add(lib)
					continue
				domains.add(domain)
			index[str(task_id)] = domains

	return index, unknown_libs


def compute_label(payload: Dict[str, object]) -> bool:
	features = payload.get("features", {})
	if not isinstance(features, dict):
		return False
	for feat in features.values():
		if not isinstance(feat, dict):
			continue
		is_correct = feat.get("is_correct", False)
		try:
			if bool(is_correct):
				return True
		except Exception:  # pylint: disable=broad-except
			continue
	return False


def resolve_sample_id(payload: Dict[str, object], fallback_name: str) -> str:
	sample_id = payload.get("sample_id")
	if isinstance(sample_id, str) and sample_id.strip():
		return sample_id
	return fallback_name


def scan_feature_files(
	data_dir: Path,
	splits: Sequence[str],
	index: Dict[str, Set[str]],
) -> List[FileRecord]:
	records: List[FileRecord] = []
	for split_name in splits:
		split_dir = data_dir / split_name
		if not split_dir.exists():
			LOGGER.warning("Split dir %s not found; skipping", split_dir)
			continue
		for path in sorted(split_dir.glob("*.pt")):
			try:
				payload = torch.load(path, map_location="cpu")
			except Exception as exc:  # pylint: disable=broad-except
				LOGGER.warning("Failed to load %s: %s", path, exc)
				continue
			base = path.stem.split("__", 1)[0]
			sample_id = resolve_sample_id(payload if isinstance(payload, dict) else {}, base)
			domains = index.get(sample_id, set())
			label = compute_label(payload if isinstance(payload, dict) else {})
			records.append(FileRecord(path=path, sample_id=sample_id, label=label, domains=domains))
	return records


def copy_records(records: Iterable[FileRecord], output_dir: Path) -> int:
	count = 0
	output_dir.mkdir(parents=True, exist_ok=True)
	for rec in records:
		dest = output_dir / rec.path.name
		shutil.copy2(rec.path, dest)
		count += 1
	return count


def main() -> None:
	args = parse_args()
	setup_logging(args.log_level)

	lib_map_path = Path(__file__).with_name("bcb_domains.json")
	if not lib_map_path.exists():
		raise FileNotFoundError(f"Library-domain mapping not found: {lib_map_path}")

	lib_to_domain = load_domain_map(lib_map_path)
	index, unknown_libs = build_task_domain_index(
		args.dataset_name,
		args.dataset_config,
		args.dataset_revision,
		lib_to_domain,
	)
	if unknown_libs:
		LOGGER.info("Encountered %s unmapped libs (first 20 shown): %s", len(unknown_libs), sorted(unknown_libs)[:20])

	data_dir = args.data_dir.expanduser().resolve()
	if not data_dir.exists():
		raise FileNotFoundError(f"Data dir not found: {data_dir}")

	output_dir = args.output_dir
	if output_dir is None:
		sanitized = args.target_domain.strip().replace(" ", "_").lower()
		output_dir = data_dir.parent / f"{data_dir.name}_domain_{sanitized}"
	output_dir = output_dir.expanduser().resolve()

	records = scan_feature_files(data_dir, args.splits, index)
	if not records:
		LOGGER.warning("No .pt files found under %s", data_dir)
		return

	_target = args.target_domain.strip().lower()
	holdout: List[FileRecord] = []
	eligible: List[FileRecord] = []
	for rec in records:
		domain_match = any(domain.lower() == _target for domain in rec.domains)
		if domain_match:
			holdout.append(rec)
		else:
			eligible.append(rec)

	_validate_ratios(0.8, 0.2)
	train_set, val_set = split_records(eligible, (0.8, 0.2), seed=args.seed)

	train_dir = output_dir / "train"
	val_dir = output_dir / "validation"
	test_dir = output_dir / "test"

	train_count = copy_records(train_set, train_dir)
	val_count = copy_records(val_set, val_dir)
	test_count = copy_records(holdout, test_dir)

	LOGGER.info(
		"Wrote %s train, %s validation, %s test to %s",
		train_count,
		val_count,
		test_count,
		output_dir,
	)


if __name__ == "__main__":
	main()