from __future__ import annotations

import argparse
import logging
import sys
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

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
	sys.path.insert(0, str(PACKAGE_PARENT))

from baselines.dataset import HiddenStateDataset
from baselines.utils import DEFAULT_DATA_ROOT, bss

LOGGER = logging.getLogger(__name__)


def _dataset_to_numpy(dataset: HiddenStateDataset, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
	upper = len(dataset) if limit is None else min(limit, len(dataset))
	feats: List[np.ndarray] = []
	labels: List[float] = []
	for idx in range(upper):
		feature_tensor, label_tensor = dataset[idx]
		feats.append(feature_tensor.cpu().numpy())
		labels.append(float(label_tensor.item()))
	if not feats:
		raise RuntimeError("Dataset yielded zero samples; adjust layer/token configuration")
	return np.stack(feats).astype(np.float32), np.asarray(labels, dtype=np.float32)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate a trained XGBoost model on cached hidden-state features")
	parser.add_argument("--model-path", type=Path, required=True, help="Path to the trained XGBoost model (.json)")
	parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory containing hidden-state .pt files")
	parser.add_argument("--test-split", default="test", help="Name of the test split directory")
	parser.add_argument("--layers", type=int, nargs="*", default=[48], help="Hidden-state layers to load")
	parser.add_argument("--token-positions", type=int, nargs="*", default=[0], help="Relative assistant token positions to load")
	parser.add_argument("--filter-empty", action="store_true", help="Filter out candidates with empty code token spans")
	parser.add_argument(
		"--combiner",
		choices=["concat", "average"],
		default="concat",
		help="How to combine layer/token vectors before feeding XGBoost",
	)
	parser.add_argument("--limit-test", type=int, help="Optional cap on the number of test samples to load")
	return parser.parse_args()


def _prepare_dataset(
	path: Path,
	split: str,
	layers: Sequence[int],
	token_positions: Sequence[int],
	filter_empty: bool,
	combiner: str,
) -> HiddenStateDataset:
	return HiddenStateDataset(
		root=path,
		split=split,
		layers=layers,
		token_positions=token_positions,
		filter_empty=filter_empty,
		combiner=combiner,
	)


def _predict(booster: xgb.Booster, matrix: xgb.DMatrix) -> np.ndarray:
	if hasattr(booster, "best_iteration") and booster.best_iteration is not None and booster.best_iteration >= 0:
		return booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
	return booster.predict(matrix)


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
	preds = (probs >= 0.5).astype(np.float32)
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
			metrics["f1"] = float(f1_score(y_true, preds))
		except ValueError:
			metrics["f1"] = float("nan")
	else:
		metrics["f1"] = float("nan")

	return metrics


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	args = _parse_args()

	if not args.layers:
		raise ValueError("At least one layer must be specified")
	if not args.token_positions:
		raise ValueError("At least one token position must be specified")

	layers = tuple(sorted(set(args.layers)))
	token_positions = tuple(sorted(set(args.token_positions)))

	LOGGER.info("Loading test dataset from %s", args.data_root)
	test_dataset = _prepare_dataset(
		args.data_root,
		args.test_split,
		layers,
		token_positions,
		args.filter_empty,
		args.combiner,
	)

	LOGGER.info("Converting test split (%d samples) to numpy", len(test_dataset))
	x_test, y_test = _dataset_to_numpy(test_dataset, args.limit_test)
	dtest = xgb.DMatrix(x_test)

	LOGGER.info("Loading booster from %s", args.model_path)
	booster = xgb.Booster()
	booster.load_model(args.model_path)

	probs = _predict(booster, dtest)
	metrics = _compute_metrics(y_test, probs)

	LOGGER.info("Test metrics: %s", metrics)
	for name, value in metrics.items():
		print(f"{name}: {value:.6f}")


if __name__ == "__main__":
	main()