from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
	import xgboost as xgb
except ImportError as exc:
	raise ImportError("xgboost must be installed to run this script") from exc

from .dataset import LineLevelHiddenStateDataset
from .utils import DEFAULT_DATA_ROOT


LOGGER = logging.getLogger(__name__)
TOP_K_VALUES: Tuple[int, ...] = (1, 2, 3)
THRESHOLD_GRID: Tuple[float, ...] = tuple(i / 10.0 for i in range(1, 10))


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Evaluate a line-level XGBoost model for bug-line localization with threshold-optimized Hit@K."
		)
	)
	parser.add_argument("--model-path", type=Path, required=True, help="Path to trained XGBoost model (.json)")
	parser.add_argument(
		"--data-root",
		type=Path,
		default=DEFAULT_DATA_ROOT,
		help="Root directory containing hidden-state .pt files grouped by split",
	)
	parser.add_argument("--test-jsonl", type=Path, required=True, help="Filtered JSONL for test samples")
	parser.add_argument(
		"--layers",
		type=int,
		nargs="*",
		default=[48],
		help="Hidden-state layers to load from prefill features",
	)
	parser.add_argument(
		"--feature-splits",
		nargs="*",
		default=("train", "validation", "test"),
		help="Feature split directories to scan under --data-root",
	)
	parser.add_argument(
		"--line-representation",
		default="last_token",
		choices=["last_token", "aggregate"],
		help="Line representation mode: last code token or aggregated non-ws/comment token states",
	)
	parser.add_argument(
		"--line-aggregation",
		default="mean",
		choices=["mean", "max"],
		help="Aggregation used when --line-representation=aggregate",
	)
	parser.add_argument("--limit-test-lines", type=int, help="Optional cap on number of test line samples to load")
	return parser.parse_args()


def _prepare_dataset(args: argparse.Namespace) -> LineLevelHiddenStateDataset:
	if not args.layers:
		raise ValueError("At least one layer must be specified")
	layers = tuple(sorted(set(int(layer) for layer in args.layers)))
	feature_splits = tuple(args.feature_splits)

	return LineLevelHiddenStateDataset(
		root=args.data_root,
		jsonl_path=args.test_jsonl,
		layers=layers,
		feature_splits=feature_splits,
		line_representation=args.line_representation,
		aggregation=args.line_aggregation,
	)


def _dataset_to_numpy_with_meta(
	dataset: LineLevelHiddenStateDataset,
	limit_lines: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str, int]]]:
	upper = len(dataset) if limit_lines is None else min(limit_lines, len(dataset))
	features: List[np.ndarray] = []
	labels: List[float] = []
	line_meta: List[Tuple[str, str, int]] = []

	for idx in range(upper):
		feature_tensor, label_tensor, metadata = dataset[idx]
		flat = feature_tensor.cpu().numpy().reshape(-1)
		features.append(flat)
		labels.append(float(label_tensor.item()))

		item = dataset._items[idx]  # pylint: disable=protected-access
		path_key = str(item.path)
		candidate_key = str(item.candidate_key)
		line_number_raw = metadata.get("line_number")
		line_number = int(line_number_raw.item()) if line_number_raw is not None else int(item.line_number)
		line_meta.append((path_key, candidate_key, line_number))

	if not features:
		raise RuntimeError("Dataset yielded zero line samples; verify JSONL/features alignment")

	return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.float32), line_meta


def _predict(booster: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
	if hasattr(booster, "best_iteration") and booster.best_iteration is not None and booster.best_iteration >= 0:
		return booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
	return booster.predict(matrix)


def _compute_hit_at_k(
	line_meta: Sequence[Tuple[str, str, int]],
	line_labels: np.ndarray,
	line_scores: np.ndarray,
	threshold: float,
	k_values: Sequence[int] = TOP_K_VALUES,
) -> Dict[str, float]:
	if len(line_meta) != int(line_scores.shape[0]) or len(line_meta) != int(line_labels.shape[0]):
		raise ValueError("line_meta, line_labels and line_scores size mismatch")
	if not k_values:
		raise ValueError("k_values must not be empty")

	unique_k_values = tuple(sorted(set(int(k) for k in k_values)))
	if any(k <= 0 for k in unique_k_values):
		raise ValueError("All k values must be positive integers")

	grouped: Dict[Tuple[str, str], List[int]] = {}
	for idx, (path_key, candidate_key, _) in enumerate(line_meta):
		grouped.setdefault((path_key, candidate_key), []).append(idx)

	hits_by_k = {k: 0 for k in unique_k_values}
	misses_by_k = {k: 0 for k in unique_k_values}
	eligible_candidates = 0
	excluded_no_incorrect = 0

	max_k = unique_k_values[-1]

	for _, indices in grouped.items():
		has_incorrect = any(line_labels[idx] > threshold for idx in indices)
		if not has_incorrect:
			excluded_no_incorrect += 1
			continue
		eligible_candidates += 1

		# Stable ranking: score descending, then line number ascending for deterministic ties.
		ranked_indices = sorted(indices, key=lambda idx: (-float(line_scores[idx]), int(line_meta[idx][2])))
		top_indices = ranked_indices[: min(max_k, len(ranked_indices))]

		is_positive_top = [line_labels[idx] > threshold for idx in top_indices]
		prefix_hits = np.cumsum(np.asarray(is_positive_top, dtype=np.int32))

		for k in unique_k_values:
			effective_k = min(k, len(top_indices))
			has_hit = bool(prefix_hits[effective_k - 1] > 0)
			if has_hit:
				hits_by_k[k] += 1
			else:
				misses_by_k[k] += 1

	metrics: Dict[str, float] = {
		"eligible_candidates": float(eligible_candidates),
		"excluded_no_incorrect_lines": float(excluded_no_incorrect),
		"candidates_total": float(len(grouped)),
		"line_samples": float(len(line_meta)),
	}
	for k in unique_k_values:
		hit_rate = float(hits_by_k[k] / eligible_candidates) if eligible_candidates > 0 else float("nan")
		metrics[f"hit_at_{k}"] = hit_rate
		metrics[f"hits_at_{k}"] = float(hits_by_k[k])
		metrics[f"misses_at_{k}"] = float(misses_by_k[k])
	return metrics


def _optimize_threshold_for_hit_at_1(
	line_meta: Sequence[Tuple[str, str, int]],
	line_labels: np.ndarray,
	line_scores: np.ndarray,
	thresholds: Sequence[float] = THRESHOLD_GRID,
	k_values: Sequence[int] = TOP_K_VALUES,
) -> Tuple[float, Dict[str, float]]:
	if not thresholds:
		raise ValueError("thresholds must not be empty")

	best_threshold = float(thresholds[0])
	best_metrics = _compute_hit_at_k(line_meta, line_labels, line_scores, threshold=best_threshold, k_values=k_values)
	best_hit_at_1 = best_metrics.get("hit_at_1", float("nan"))

	for threshold in thresholds[1:]:
		threshold = float(threshold)
		metrics = _compute_hit_at_k(line_meta, line_labels, line_scores, threshold=threshold, k_values=k_values)
		hit_at_1 = metrics.get("hit_at_1", float("nan"))
		if hit_at_1 > best_hit_at_1:
			best_threshold = threshold
			best_metrics = metrics
			best_hit_at_1 = hit_at_1

	best_metrics["best_threshold"] = best_threshold
	return best_threshold, best_metrics


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	args = _parse_args()

	LOGGER.info("Loading line dataset from feature root %s", args.data_root)
	test_dataset = _prepare_dataset(args)
	LOGGER.info("test lines=%d feature_dim=%s", len(test_dataset), test_dataset.feature_dim)
	LOGGER.info("test stats=%s", test_dataset.stats)

	LOGGER.info("Converting test dataset (%d lines) to numpy", len(test_dataset))
	x_test, y_test, line_meta = _dataset_to_numpy_with_meta(test_dataset, args.limit_test_lines)
	dtest = xgb.DMatrix(x_test)

	LOGGER.info("Loading booster from %s", args.model_path)
	booster = xgb.Booster()
	booster.load_model(args.model_path)

	line_scores = _predict(booster, dtest)

	best_threshold, metrics = _optimize_threshold_for_hit_at_1(line_meta, y_test, line_scores)
	LOGGER.info("Best threshold by Hit@1 from [0.1..0.9]: %.1f", best_threshold)
	LOGGER.info("Top-k hit metrics at best threshold: %s", metrics)

	output_keys = ("best_threshold", "hit_at_1", "hit_at_2", "hit_at_3")
	for name in output_keys:
		value = metrics.get(name)
		if value is not None:
			print(f"{name}: {value:.6f}")


if __name__ == "__main__":
	main()