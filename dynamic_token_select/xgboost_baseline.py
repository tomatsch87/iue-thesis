"""
Train an XGBoost baseline on fixed-length flattened sequence features.

Default training behavior:
- Loads `train` and `validation` splits from `--data-root`.
- Uses layer `[48]` and token filter `top_k-token-entropy` by default.
- Applies `filter_empty=True` unless `--keep-empty` is passed.
- Requires `--entropy-k` to enforce fixed token count (feature length consistency).
- Saves the best booster to `outputs/xgboost/model.json`.

Important options:
- Data/filter controls: `--layers`, `--token-filter`, `--token-filter-config`,
  `--entropy-k`, `--filter-empty/--keep-empty`.
- XGBoost knobs: `--learning-rate`, `--max-depth`, `--min-child-weight`,
  `--subsample`, `--colsample-bytree`, `--reg-lambda`, `--reg-alpha`, `--tree-method`.
- Search/control: `--grid-search-learning-rate`, `--num-boost-round`,
  `--early-stopping-rounds`, `--report-every`, `--limit-train`, `--limit-valid`.
- Imbalance handling: `--use-class-weights`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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

from dynamic_token_select.dataset import HiddenStateSequenceDataset
from dynamic_token_select.utils import DEFAULT_DATA_ROOT, bss, seed_everything

LOGGER = logging.getLogger(__name__)

LEARNING_RATE_GRID_FACTORS: Tuple[float, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
)

TOP_K_FILTERS: Tuple[str, ...] = (
    "top_k-token-entropy",
    "new-line+top_k-token-entropy",
    "comment-or-identifier+top_k-token-entropy",
)


def _score_metrics(metrics: Dict[str, float]) -> float:
    roc_auc = metrics.get("roc_auc")
    if roc_auc is not None and not math.isnan(roc_auc):
        return roc_auc
    return metrics.get("accuracy", float("-inf"))


def _dataset_to_numpy(dataset: HiddenStateSequenceDataset, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    upper = len(dataset) if limit is None else min(limit, len(dataset))
    feats: List[np.ndarray] = []
    labels: List[float] = []
    expected_tokens: Optional[int] = None
    for idx in range(upper):
        feature_tensor, label_tensor = dataset[idx]
        if feature_tensor.dim() < 2:
            raise ValueError(
                "Hidden state sequence must have shape [tokens, layers, hidden_size] or similar"
            )
        token_count = int(feature_tensor.size(0))
        if expected_tokens is None:
            expected_tokens = token_count
        elif token_count != expected_tokens:
            raise ValueError(
                f"Inconsistent token count: expected {expected_tokens}, got {token_count} at index {idx}"
            )
        flat = feature_tensor.reshape(-1).cpu().numpy()
        feats.append(flat)
        labels.append(float(label_tensor.item()))
    if not feats:
        raise RuntimeError("Dataset yielded zero samples; adjust filter/k configuration")
    feature_matrix = np.stack(feats).astype(np.float32)
    label_vector = np.asarray(labels, dtype=np.float32)
    LOGGER.info("Flattened features to shape %s (tokens=%s)", feature_matrix.shape, expected_tokens)
    return feature_matrix, label_vector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an XGBoost baseline on cached hidden-state features")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root directory containing hidden-state .pt files")
    parser.add_argument("--train-split", default="train", help="Name of the training split directory")
    parser.add_argument("--valid-split", default="validation", help="Name of the validation split directory")
    parser.add_argument("--layers", type=int, nargs="*", default=[48], help="Hidden-state layers to load (all available if omitted)")
    parser.add_argument(
        "--filter-empty",
        action="store_true",
        default=True,
        help="Filter out candidates with empty code token spans (dataset default)",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_false",
        dest="filter_empty",
        help="Disable empty-span filtering",
    )
    parser.add_argument(
        "--token-filter",
        choices=TOP_K_FILTERS,
        default="top_k-token-entropy",
        help="Token filter applied by HiddenStateSequenceDataset (restricted to top-k entropy filters)",
    )
    parser.add_argument(
        "--token-filter-config",
        type=str,
        default="{}",
        help="JSON-encoded dictionary of additional config options for the token filter",
    )
    parser.add_argument(
        "--entropy-k",
        type=int,
        required=True,
        help="Number of tokens to keep via top-k entropy selection; determines feature length",
    )
    parser.add_argument("--num-boost-round", type=int, default=500, help="Maximum boosting iterations")
    parser.add_argument("--early-stopping-rounds", type=int, default=50, help="Stop if no validation improvement for this many rounds (<=0 disables)")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="XGBoost learning rate (eta)")
    parser.add_argument(
        "--grid-search-learning-rate",
        action="store_true",
        help="Enable grid search over the learning rate using preset multiplicative factors around the provided value",
    )
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth")
    parser.add_argument("--min-child-weight", type=float, default=1.0, help="Minimum sum of instance weight needed in a child")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsampling ratio")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsampling ratio per tree")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization term on weights")
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularization term on weights")
    parser.add_argument("--tree-method", default="auto", choices=["auto", "hist", "approx", "gpu_hist"], help="Tree construction algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for numpy/xgboost")
    parser.add_argument("--limit-train", type=int, help="Optional cap on the number of training samples to load")
    parser.add_argument("--limit-valid", type=int, help="Optional cap on the number of validation samples to load")
    parser.add_argument("--use-class-weights", action="store_true", help="Enable inverse-frequency class weights during training")
    parser.add_argument("--model-path", type=Path, default=Path("outputs/xgboost/model.json"), help="Where to save the trained booster")
    parser.add_argument("--report-every", type=int, default=10, help="Verbose evaluation frequency (0 disables periodic eval prints)")
    return parser.parse_args()


def _prepare_dataset(
    path: Path,
    split: str,
    layers: Optional[Sequence[int]],
    token_filter: str,
    token_filter_config: Mapping[str, object],
) -> HiddenStateSequenceDataset:
    return HiddenStateSequenceDataset(
        root=path,
        split=split,
        layers=layers,
        token_filter=token_filter,
        token_filter_config=token_filter_config,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    if not args.layers:
        raise ValueError("At least one layer must be specified")
    layers = tuple(sorted(set(args.layers))) if args.layers else None
    if args.token_filter not in TOP_K_FILTERS:
        raise ValueError(f"Token filter {args.token_filter} is not supported; choose a top-k entropy filter")
    if args.entropy_k is None or args.entropy_k <= 0:
        raise ValueError("--entropy-k must be a positive integer")

    token_filter_config: Dict[str, object]
    try:
        parsed = json.loads(args.token_filter_config)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--token-filter-config must be valid JSON (example: '{\"add_last_token\": true}')"
        ) from exc
    if parsed is None:
        token_filter_config = {}
    elif isinstance(parsed, dict):
        token_filter_config = dict(parsed)
    else:
        raise ValueError("--token-filter-config must decode to a JSON object (dictionary)")

    token_filter_config["k"] = int(args.entropy_k)

    LOGGER.info("Loading datasets from %s", args.data_root)
    train_dataset = _prepare_dataset(
        args.data_root,
        args.train_split,
        layers,
        args.token_filter,
        token_filter_config,
    )
    valid_dataset = _prepare_dataset(
        args.data_root,
        args.valid_split,
        layers,
        args.token_filter,
        token_filter_config,
    )

    seed_everything(args.seed)

    LOGGER.info("Converting training split (%d samples) to numpy", len(train_dataset))
    x_train, y_train = _dataset_to_numpy(train_dataset, args.limit_train)
    LOGGER.info("Converting validation split (%d samples) to numpy", len(valid_dataset))
    x_valid, y_valid = _dataset_to_numpy(valid_dataset, args.limit_valid)

    train_weights: Optional[np.ndarray] = None
    if args.use_class_weights:
        pos_count = float(np.count_nonzero(y_train == 1.0))
        neg_count = float(len(y_train) - pos_count)
        if pos_count > 0.0 and neg_count > 0.0:
            pos_weight = neg_count / pos_count
            train_weights = np.where(y_train == 1.0, pos_weight, 1.0).astype(np.float32)
            LOGGER.info("Applying class weights with positive weight %.4f", pos_weight)
        else:
            LOGGER.warning(
                "Skipping class weights because dataset contains only one class (pos=%d, neg=%d)",
                int(pos_count),
                int(neg_count),
            )

    dtrain = xgb.DMatrix(x_train, label=y_train, weight=train_weights)
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
                metrics["f1"] = float(f1_score(y_valid, val_preds))
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
            raise ValueError("Learning rate grid search produced no positive candidates; adjust --learning-rate")
        LOGGER.info(
            "Running learning rate grid search with candidates: %s",
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
        "Saved best booster (learning rate %.5f, score %.5f) to %s with metrics %s",
        best_lr,
        best_score,
        args.model_path,
        best_metrics,
    )


if __name__ == "__main__":
    main()