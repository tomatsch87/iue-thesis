"""
Train a token-level XGBoost baseline from hidden-state features.

Default training behavior:
- Requires `--train-jsonl` and `--valid-jsonl` aligned with feature files.
- Loads layers `[48]` from feature splits `train/validation/test` by default.
- Converts each token sample to a flattened feature vector.
- Trains binary logistic XGBoost with class-imbalance handling via `scale_pos_weight`
	(auto ratio if not explicitly set).
- Saves to `outputs/token_level/xgboost/model.json`.

Important options:
- Data controls: `--layers`, `--feature-splits`, `--limit-train`, `--limit-valid`.
- XGBoost knobs: `--learning-rate`, `--max-depth`, `--min-child-weight`,
	`--subsample`, `--colsample-bytree`, `--reg-lambda`, `--reg-alpha`, `--tree-method`.
- Search/control: `--grid-search-learning-rate`, `--num-boost-round`,
	`--early-stopping-rounds`, `--report-every`, `--scale-pos-weight`.
"""

from __future__ import annotations

import argparse
import logging
import math
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
from .utils import DEFAULT_DATA_ROOT, bss, seed_everything


LOGGER = logging.getLogger(__name__)

LEARNING_RATE_GRID_FACTORS: Tuple[float, ...] = (
	0.8,
	0.9,
	1.0,
	1.1,
	1.2,
)


def _score_metrics(metrics: Dict[str, float]) -> float:
	roc_auc = metrics.get("roc_auc")
	if roc_auc is not None and not math.isnan(roc_auc):
		return roc_auc
	return metrics.get("accuracy", float("-inf"))


def _dataset_to_numpy(
	dataset: TokenLevelHiddenStateDataset,
	limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
	upper = len(dataset) if limit is None else min(limit, len(dataset))
	feats: List[np.ndarray] = []
	labels: List[float] = []
	for idx in range(upper):
		feature_tensor, label_tensor, _ = dataset[idx]
		flat = feature_tensor.cpu().numpy().reshape(-1)
		feats.append(flat)
		labels.append(float(label_tensor.item()))
	if not feats:
		raise RuntimeError("Dataset yielded zero token samples; verify JSONL/features alignment")
	return np.stack(feats).astype(np.float32), np.asarray(labels, dtype=np.float32)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train an XGBoost token-level hallucination detector on cached hidden-state features",
	)
	parser.add_argument(
		"--data-root",
		type=Path,
		default=DEFAULT_DATA_ROOT,
		help="Root directory containing hidden-state .pt files grouped by split directories",
	)
	parser.add_argument("--train-jsonl", type=Path, required=True, help="Filtered JSONL for training samples")
	parser.add_argument("--valid-jsonl", type=Path, required=True, help="Filtered JSONL for validation samples")
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
		"--num-boost-round",
		"--n-estimators",
		dest="num_boost_round",
		type=int,
		default=3000,
		help="Maximum boosting iterations (alias: --n-estimators)",
	)
	parser.add_argument(
		"--early-stopping-rounds",
		type=int,
		default=300,
		help="Stop if no validation improvement for this many rounds (<=0 disables)",
	)
	parser.add_argument("--learning-rate", type=float, default=0.02, help="XGBoost learning rate (eta)")
	parser.add_argument(
		"--grid-search-learning-rate",
		action="store_true",
		help="Enable grid search over learning-rate factors around --learning-rate",
	)
	parser.add_argument("--max-depth", type=int, default=4, help="Maximum tree depth")
	parser.add_argument(
		"--min-child-weight",
		type=float,
		default=10.0,
		help="Minimum sum of instance weight needed in a child",
	)
	parser.add_argument(
		"--subsample", type=float, default=0.8, help="Row subsampling ratio")
	parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsampling ratio per tree")
	parser.add_argument("--reg-lambda", type=float, default=2.0, help="L2 regularization term on weights")
	parser.add_argument("--reg-alpha", type=float, default=0.1, help="L1 regularization term on weights")
	parser.add_argument(
		"--tree-method",
		default="auto",
		choices=["auto", "hist", "approx", "gpu_hist"],
		help="Tree construction algorithm",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed for numpy/xgboost")
	parser.add_argument("--limit-train", type=int, help="Optional cap on the number of training token samples")
	parser.add_argument("--limit-valid", type=int, help="Optional cap on the number of validation token samples")
	parser.add_argument(
		"--scale-pos-weight",
		type=float,
		help="Positive-class weight for imbalance. If unset, uses train_neg/train_pos ratio",
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=Path("outputs/token_level/xgboost/model.json"),
		help="Where to save the trained booster",
	)
	parser.add_argument(
		"--report-every",
		type=int,
		default=10,
		help="Verbose evaluation frequency (0 disables periodic eval prints)",
	)
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


def _log_dataset_stats(prefix: str, dataset: TokenLevelHiddenStateDataset) -> None:
	LOGGER.info("%s samples=%d feature_dim=%s", prefix, len(dataset), dataset.feature_dim)
	LOGGER.info("%s stats=%s", prefix, dataset.stats)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	args = _parse_args()

	if not args.layers:
		raise ValueError("At least one layer must be specified")
	layers = tuple(sorted(set(int(layer) for layer in args.layers)))
	feature_splits = tuple(args.feature_splits)

	LOGGER.info("Loading token datasets from feature root %s", args.data_root)
	train_dataset = _prepare_dataset(
		root=args.data_root,
		jsonl_path=args.train_jsonl,
		layers=layers,
		feature_splits=feature_splits,
	)
	valid_dataset = _prepare_dataset(
		root=args.data_root,
		jsonl_path=args.valid_jsonl,
		layers=layers,
		feature_splits=feature_splits,
	)

	_log_dataset_stats("train", train_dataset)
	_log_dataset_stats("validation", valid_dataset)

	seed_everything(args.seed)

	LOGGER.info("Converting training dataset (%d samples) to numpy", len(train_dataset))
	x_train, y_train = _dataset_to_numpy(train_dataset, args.limit_train)
	LOGGER.info("Converting validation dataset (%d samples) to numpy", len(valid_dataset))
	x_valid, y_valid = _dataset_to_numpy(valid_dataset, args.limit_valid)
	LOGGER.info(
		"Label stats (1=hallucination): train_pos=%d/%d valid_pos=%d/%d",
		int(np.count_nonzero(y_train == 1.0)),
		len(y_train),
		int(np.count_nonzero(y_valid == 1.0)),
		len(y_valid),
	)

	pos_count = float(np.count_nonzero(y_train == 1.0))
	neg_count = float(len(y_train) - pos_count)
	if args.scale_pos_weight is not None:
		scale_pos_weight = float(args.scale_pos_weight)
		if scale_pos_weight <= 0.0:
			raise ValueError("--scale-pos-weight must be > 0")
	elif pos_count > 0.0 and neg_count > 0.0:
		scale_pos_weight = neg_count / pos_count
	else:
		scale_pos_weight = 1.0
		LOGGER.warning(
			"Using scale_pos_weight=1.0 because train split has a single class (pos=%d, neg=%d)",
			int(pos_count),
			int(neg_count),
		)
	LOGGER.info("Using scale_pos_weight=%.4f", scale_pos_weight)

	dtrain = xgb.DMatrix(x_train, label=y_train)
	dvalid = xgb.DMatrix(x_valid, label=y_valid)

	base_params = {
		"objective": "binary:logistic",
		"eval_metric": ["logloss", "auc"],
		"max_depth": args.max_depth,
		"min_child_weight": args.min_child_weight,
		"subsample": args.subsample,
		"colsample_bytree": args.colsample_bytree,
		"reg_lambda": args.reg_lambda,
		"reg_alpha": args.reg_alpha,
		"scale_pos_weight": scale_pos_weight,
		"tree_method": args.tree_method,
		"seed": args.seed,
	}

	evals = [(dtrain, "train"), (dvalid, "validation")]
	early_stop = args.early_stopping_rounds if args.early_stopping_rounds and args.early_stopping_rounds > 0 else None
	verbose_eval = args.report_every if args.report_every > 0 else False

	def train_with_learning_rate(current_lr: float) -> Tuple[xgb.Booster, Dict[str, float]]:
		params = dict(base_params)
		params["eta"] = current_lr
		LOGGER.info("Training XGBoost model with params: %s", params)

		booster = xgb.train(
			params,
			dtrain,
			num_boost_round=args.num_boost_round,
			evals=evals,
			early_stopping_rounds=early_stop,
			verbose_eval=verbose_eval,
		)

		def _predict(matrix: xgb.DMatrix) -> np.ndarray:
			if hasattr(booster, "best_iteration") and booster.best_iteration is not None and booster.best_iteration >= 0:
				return booster.predict(matrix, iteration_range=(0, booster.best_iteration + 1))
			return booster.predict(matrix)

		val_probs = _predict(dvalid)
		val_preds = (val_probs >= 0.5).astype(np.float32)
		accuracy = float((val_preds == y_valid).mean())

		metrics: Dict[str, float] = {
			"accuracy": accuracy,
		}

		if np.unique(y_valid).size > 1:
			metrics["bss"] = float(bss(val_probs, y_valid))
		else:
			metrics["bss"] = float("nan")

		if roc_auc_score is not None:
			try:
				metrics["roc_auc"] = float(roc_auc_score(y_valid, val_probs))
			except ValueError:
				metrics["roc_auc"] = float("nan")
		else:
			metrics["roc_auc"] = float("nan")

		if f1_score is not None:
			try:
				metrics["f1"] = float(f1_score(y_valid, val_preds, pos_label=1))
			except ValueError:
				metrics["f1"] = float("nan")
		else:
			metrics["f1"] = float("nan")

		LOGGER.info("Validation metrics at learning rate %.5f: %s", current_lr, metrics)
		return booster, metrics

	if args.grid_search_learning_rate:
		candidate_rates = sorted(
			{
				max(1e-5, args.learning_rate * factor)
				for factor in LEARNING_RATE_GRID_FACTORS
				if args.learning_rate * factor > 0.0
			}
		)
		if not candidate_rates:
			raise ValueError("Learning-rate grid search produced no positive candidates; adjust --learning-rate")
		LOGGER.info(
			"Running learning-rate grid search with candidates: %s",
			", ".join(f"{rate:.5f}" for rate in candidate_rates),
		)
	else:
		candidate_rates = [args.learning_rate]

	best_booster: Optional[xgb.Booster] = None
	best_metrics: Optional[Dict[str, float]] = None
	best_lr: Optional[float] = None
	best_score = float("-inf")

	for current_lr in candidate_rates:
		booster, metrics = train_with_learning_rate(current_lr)
		score = _score_metrics(metrics)
		if score > best_score:
			best_score = score
			best_booster = booster
			best_metrics = metrics
			best_lr = current_lr

	if best_booster is None or best_metrics is None or best_lr is None:
		raise RuntimeError("Training did not produce any booster")

	args.model_path.parent.mkdir(parents=True, exist_ok=True)
	best_booster.save_model(args.model_path)
	LOGGER.info(
		"Saved best booster (learning_rate=%.5f, score=%.5f) to %s with metrics %s",
		best_lr,
		best_score,
		args.model_path,
		best_metrics,
	)


if __name__ == "__main__":
	main()