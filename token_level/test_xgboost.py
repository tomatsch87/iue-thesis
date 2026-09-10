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

from .dataset import TokenLevelHiddenStateDataset
from .utils import DEFAULT_DATA_ROOT, bss


LOGGER = logging.getLogger(__name__)


LineTokenMeta = Tuple[str, str, int, int]


def _dataset_to_numpy_with_meta(
	dataset: TokenLevelHiddenStateDataset,
	limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[LineTokenMeta]]:
	upper = len(dataset) if limit is None else min(limit, len(dataset))
	feats: List[np.ndarray] = []
	labels: List[float] = []
	line_meta: List[LineTokenMeta] = []

	for idx in range(upper):
		feature_tensor, label_tensor, metadata = dataset[idx]
		flat = feature_tensor.cpu().numpy().reshape(-1)
		feats.append(flat)
		labels.append(float(label_tensor.item()))

		item = dataset._items[idx]  # pylint: disable=protected-access
		path_key = str(item.path)
		candidate_key = str(item.candidate_key)

		line_number_raw = metadata.get("line_number")
		line_label_raw = metadata.get("line_label")
		line_number = int(line_number_raw.item()) if line_number_raw is not None else -1
		line_label = int(line_label_raw.item()) if line_label_raw is not None else -1
		line_meta.append((path_key, candidate_key, line_number, line_label))

	if not feats:
		raise RuntimeError("Dataset yielded zero token samples; verify JSONL/features alignment")
	return np.stack(feats).astype(np.float32), np.asarray(labels, dtype=np.float32), line_meta


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate a token-level XGBoost model on cached hidden-state features",
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
	parser.add_argument("--limit-test", type=int, help="Optional cap on the number of test token samples")
	return parser.parse_args()


def _prepare_dataset(
	root: Path,
	jsonl_path: Path,
	layers: Sequence[int],
	feature_splits: Sequence[str],
) -> TokenLevelHiddenStateDataset:
	return TokenLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=layers,
		feature_splits=feature_splits,
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


def _compute_partial_credit_line_f1(
	probs: np.ndarray,
	line_meta: Sequence[LineTokenMeta],
	threshold: float,
) -> Tuple[float, int]:
	line_state: Dict[Tuple[str, str, int], Dict[str, int | bool]] = {}
	inconsistent_line_labels = 0

	for prob, (path_key, candidate_key, line_number, line_label) in zip(probs, line_meta):
		if line_number < 0 or line_label < 0:
			continue

		line_key = (path_key, candidate_key, line_number)
		y_line_error = int(line_label)
		pred_token_hallucination = bool(prob >= threshold)

		if line_key not in line_state:
			line_state[line_key] = {
				"y_line_error": y_line_error,
				"pred_any_error": pred_token_hallucination,
			}
			continue

		entry = line_state[line_key]
		if int(entry["y_line_error"]) != y_line_error:
			inconsistent_line_labels += 1
		entry["pred_any_error"] = bool(entry["pred_any_error"]) or pred_token_hallucination

	if inconsistent_line_labels > 0:
		LOGGER.warning("Observed %d inconsistent line labels across grouped tokens", inconsistent_line_labels)

	if not line_state:
		return float("nan"), 0

	y_true_line = np.asarray([int(entry["y_line_error"]) for entry in line_state.values()], dtype=np.int32)
	y_pred_line = np.asarray([1 if bool(entry["pred_any_error"]) else 0 for entry in line_state.values()], dtype=np.int32)
	line_f1 = _binary_f1(y_true_line, y_pred_line)
	return line_f1, int(len(y_true_line))


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	args = _parse_args()

	if not args.layers:
		raise ValueError("At least one layer must be specified")

	layers = tuple(sorted(set(int(layer) for layer in args.layers)))
	feature_splits = tuple(args.feature_splits)

	LOGGER.info("Loading test token dataset from feature root %s", args.data_root)
	test_dataset = _prepare_dataset(
		root=args.data_root,
		jsonl_path=args.test_jsonl,
		layers=layers,
		feature_splits=feature_splits,
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
	line_partial_f1, line_groups = _compute_partial_credit_line_f1(probs, line_meta, best_threshold)
	metrics["line_partial_f1"] = float(line_partial_f1)
	metrics["line_groups_used"] = float(line_groups)

	LOGGER.info("Best threshold by F1 from [0.1..0.9]: %.1f", best_threshold)
	LOGGER.info("Test metrics at best threshold: %s", metrics)
	for name, value in metrics.items():
		print(f"{name}: {value:.6f}")


if __name__ == "__main__":
	main()