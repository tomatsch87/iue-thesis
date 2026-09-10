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

try:
	from sklearn.metrics import f1_score, roc_auc_score
except ImportError:
	f1_score = None
	roc_auc_score = None

from .dataset import LineLevelHiddenStateDataset
from .utils import DEFAULT_DATA_ROOT, bss


LOGGER = logging.getLogger(__name__)


LineMeta = Tuple[str, str, int]


def _dataset_to_numpy_with_meta(
	dataset: LineLevelHiddenStateDataset,
	limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[LineMeta]]:
	upper = len(dataset) if limit is None else min(limit, len(dataset))
	feats: List[np.ndarray] = []
	labels: List[float] = []
	line_meta: List[LineMeta] = []

	for idx in range(upper):
		feature_tensor, label_tensor, metadata = dataset[idx]
		flat = feature_tensor.cpu().numpy().reshape(-1)
		feats.append(flat)
		labels.append(float(label_tensor.item()))

		item = dataset._items[idx]  # pylint: disable=protected-access
		path_key = str(item.path)
		candidate_key = str(item.candidate_key)
		line_number_raw = metadata.get("line_number")
		line_number = int(line_number_raw.item()) if line_number_raw is not None else -1
		line_meta.append((path_key, candidate_key, line_number))

	if not feats:
		raise RuntimeError("Dataset yielded zero line samples; verify JSONL/features alignment")
	return np.stack(feats).astype(np.float32), np.asarray(labels, dtype=np.float32), line_meta


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate a line-level XGBoost model on cached hidden-state features",
	)
	parser.add_argument("--model-path", type=Path, required=True, help="Path to trained XGBoost model (.json)")
	parser.add_argument(
		"--data-root",
		type=Path,
		default=DEFAULT_DATA_ROOT,
		help="Root directory containing hidden-state .pt files grouped by split directories",
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
	parser.add_argument("--limit-test", type=int, help="Optional cap on the number of test line samples")
	return parser.parse_args()


def _prepare_dataset(
	root: Path,
	jsonl_path: Path,
	layers: Sequence[int],
	feature_splits: Sequence[str],
	line_representation: str,
	aggregation: str,
) -> LineLevelHiddenStateDataset:
	return LineLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=layers,
		feature_splits=feature_splits,
		line_representation=line_representation,
		aggregation=aggregation,
	)


def _predict(booster: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
	if hasattr(booster, "best_iteration") and booster.best_iteration is not None and booster.best_iteration >= 0:
		return booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
	return booster.predict(matrix)


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
	fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
	fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
	denom = 2 * tp + fp + fn
	if denom == 0:
		return 0.0
	return float((2 * tp) / denom)


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
	preds = (probs >= threshold).astype(np.float32)
	metrics: Dict[str, float] = {"accuracy": float((preds == y_true).mean())}

	if np.unique(y_true).size > 1:
		metrics["bss"] = float(bss(probs, y_true))
	else:
		metrics["bss"] = float("nan")

	if roc_auc_score is not None:
		try:
			metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
		except ValueError:
			metrics["roc_auc"] = float("nan")
	else:
		metrics["roc_auc"] = float("nan")

	if f1_score is not None:
		try:
			metrics["f1"] = float(f1_score(y_true, preds, pos_label=1))
		except ValueError:
			metrics["f1"] = float("nan")
	else:
		metrics["f1"] = _binary_f1(y_true.astype(np.int32), preds.astype(np.int32))

	return metrics


def _optimize_threshold_for_f1(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
	thresholds = [i / 10.0 for i in range(1, 10)]
	best_threshold = thresholds[0]
	best_metrics = _compute_metrics(y_true, probs, best_threshold)
	best_f1 = best_metrics.get("f1", float("nan"))

	for threshold in thresholds[1:]:
		metrics = _compute_metrics(y_true, probs, threshold)
		f1 = metrics.get("f1", float("nan"))
		if np.isnan(best_f1) or (not np.isnan(f1) and f1 > best_f1):
			best_threshold = threshold
			best_metrics = metrics
			best_f1 = f1

	best_metrics = dict(best_metrics)
	best_metrics["best_threshold"] = float(best_threshold)
	return best_threshold, best_metrics


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	args = _parse_args()

	if not args.layers:
		raise ValueError("At least one layer must be specified")

	layers = tuple(sorted(set(int(layer) for layer in args.layers)))
	feature_splits = tuple(args.feature_splits)

	LOGGER.info("Loading test line dataset from feature root %s", args.data_root)
	test_dataset = _prepare_dataset(
		root=args.data_root,
		jsonl_path=args.test_jsonl,
		layers=layers,
		feature_splits=feature_splits,
		line_representation=args.line_representation,
		aggregation=args.line_aggregation,
	)

	LOGGER.info("test samples=%d feature_dim=%s", len(test_dataset), test_dataset.feature_dim)
	LOGGER.info("test stats=%s", test_dataset.stats)

	LOGGER.info("Converting test dataset (%d samples) to numpy", len(test_dataset))
	x_test, y_test, line_meta = _dataset_to_numpy_with_meta(test_dataset, args.limit_test)
	dtest = xgb.DMatrix(x_test)

	LOGGER.info("Loading booster from %s", args.model_path)
	booster = xgb.Booster()
	booster.load_model(args.model_path)

	probs = _predict(booster, dtest)
	best_threshold, metrics = _optimize_threshold_for_f1(y_test, probs)
	metrics["line_samples"] = float(len(line_meta))

	LOGGER.info("Best threshold by F1 from [0.1..0.9]: %.1f", best_threshold)
	LOGGER.info("Line-level test metrics at best threshold: %s", metrics)
	for name, value in metrics.items():
		print(f"{name}: {value:.6f}")


if __name__ == "__main__":
	main()