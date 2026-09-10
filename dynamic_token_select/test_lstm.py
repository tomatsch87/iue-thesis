from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset

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
from dynamic_token_select.lstm import DynamicTokenLSTM, LSTMConfig
from dynamic_token_select.utils import DEVICE, bss

LOGGER = logging.getLogger(__name__)


def _json_object(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON for token filter config: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Token filter config must be a JSON object.")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LSTM on HiddenStateSequenceDataset")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to the saved checkpoint (.pt)")
    parser.add_argument("--test-split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--data-root", type=Path, help="Optional override for the data root; defaults to checkpoint config")
    parser.add_argument("--layers", type=int, nargs="*", help="Hidden-state layers to load (defaults to checkpoint config)")
    parser.add_argument(
        "--filter-empty",
        dest="filter_empty",
        action="store_true",
        help="Filter out candidates with empty code token spans",
    )
    parser.add_argument(
        "--no-filter-empty",
        dest="filter_empty",
        action="store_false",
        help="Keep candidates with empty code token spans",
    )
    parser.set_defaults(filter_empty=None)
    parser.add_argument("--token-filter", help="Optional token filter strategy override")
    parser.add_argument("--token-filter-config", type=_json_object, help="JSON string with token filter parameters override")
    parser.add_argument("--tokenizer-name", help="Tokenizer override used by token filters")
    parser.add_argument("--batch-size", type=int, help="Evaluation batch size (defaults to checkpoint config)")
    parser.add_argument("--num-workers", type=int, help="Dataloader workers (defaults to checkpoint config)")
    parser.add_argument("--limit-test", type=int, help="Optional cap on the number of test samples to load")
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint should be a dict with 'state_dict' and 'config'")
    if "state_dict" not in checkpoint or "config" not in checkpoint:
        raise RuntimeError("Checkpoint missing required keys 'state_dict' and/or 'config'")
    return checkpoint["state_dict"], checkpoint["config"]


def _prepare_config(saved_config: Dict[str, Any], overrides: argparse.Namespace) -> LSTMConfig:
    base_cfg = LSTMConfig()
    cfg = LSTMConfig.from_wandb_config(base_cfg, saved_config)
    if overrides.data_root is not None:
        cfg.data_root = overrides.data_root.resolve()
    if overrides.layers is not None:
        cfg.layers = tuple(sorted(set(overrides.layers))) if overrides.layers else None
    if overrides.filter_empty is not None:
        cfg.filter_empty = bool(overrides.filter_empty)
    if overrides.token_filter is not None:
        cfg.token_filter = str(overrides.token_filter).lower()
    if overrides.token_filter_config is not None:
        cfg.token_filter_config = dict(overrides.token_filter_config)
    if overrides.tokenizer_name is not None:
        cfg.tokenizer_name = overrides.tokenizer_name
    if overrides.batch_size is not None:
        cfg.batch_size = int(overrides.batch_size)
    if overrides.num_workers is not None:
        cfg.num_workers = int(overrides.num_workers)
    return cfg


def _dataset_from_config(
    config: LSTMConfig,
    split: str,
) -> HiddenStateSequenceDataset:
    return HiddenStateSequenceDataset(
        root=config.data_root,
        split=split,
        layers=config.layers,
        filter_empty=config.filter_empty,
        token_filter=config.token_filter,
        token_filter_config=config.token_filter_config,
        tokenizer_name=config.tokenizer_name,
    )


def _make_sequence_collate() -> Any:
    def _collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences, labels = zip(*batch)
        flattened = [seq.reshape(seq.size(0), -1).to(torch.float32) for seq in sequences]
        lengths = torch.tensor([seq.size(0) for seq in flattened], dtype=torch.long)
        padded = pad_sequence(flattened, batch_first=True, padding_value=0.0)
        stacked_labels = torch.stack(labels).to(torch.float32)
        return padded, lengths, stacked_labels

    return _collate


def _build_dataloader(config: LSTMConfig, dataset: HiddenStateSequenceDataset, limit: Optional[int]) -> DataLoader:
    eval_dataset = dataset
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit-test must be > 0")
        eval_dataset = Subset(dataset, list(range(min(limit, len(dataset)))))

    return DataLoader(
        eval_dataset,
        batch_size=max(int(config.batch_size), 1),
        shuffle=False,
        drop_last=False,
        num_workers=max(int(config.num_workers), 0),
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=_make_sequence_collate(),
    )


def _compute_roc_auc(labels: Sequence[float], probs: Sequence[float]) -> float:
    if roc_auc_score is None:
        return float("nan")
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


def _compute_f1(labels: Sequence[float], preds: Sequence[float]) -> float:
    if f1_score is None:
        return float("nan")
    try:
        return float(f1_score(labels, preds))
    except ValueError:
        return float("nan")


def _evaluate(model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, float]:
    model.eval()
    losses: list[float] = []
    all_labels: list[float] = []
    all_probs: list[float] = []
    all_preds: list[float] = []
    correct = 0
    total = 0

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, lengths, labels in data_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics: Dict[str, float] = {
        "loss": sum(losses) / max(len(losses), 1),
        "accuracy": correct / max(total, 1),
    }

    labels_array = np.asarray(all_labels, dtype=np.float32)
    probs_array = np.asarray(all_probs, dtype=np.float32)
    if labels_array.size > 0 and np.unique(labels_array).size > 1:
        metrics["bss"] = float(bss(probs_array, labels_array))
    else:
        metrics["bss"] = float("nan")

    metrics["roc_auc"] = _compute_roc_auc(all_labels, all_probs)
    metrics["f1"] = _compute_f1(all_labels, all_preds)
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = _parse_args()

    state_dict, saved_config = _load_checkpoint(args.checkpoint_path)
    config = _prepare_config(saved_config, args)

    LOGGER.info("Evaluating LSTM checkpoint %s on split '%s'", args.checkpoint_path, args.test_split)
    LOGGER.info("Resolved config: %s", config)

    if config.input_dim <= 0:
        raise ValueError("Invalid input_dim in checkpoint config")

    dataset = _dataset_from_config(config, args.test_split)
    if dataset.layer_count is None or dataset.hidden_size is None:
        raise RuntimeError("Dataset metadata incomplete; could not infer flattened token feature dimension")
    dataset_input_dim = int(dataset.layer_count) * int(dataset.hidden_size)
    if dataset_input_dim != int(config.input_dim):
        raise ValueError(
            "Flattened per-token feature dimension does not match LSTM input_dim: "
            f"{dataset_input_dim} vs {config.input_dim}"
        )
    data_loader = _build_dataloader(config, dataset, args.limit_test)

    model = DynamicTokenLSTM(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.lstm_layers,
        dropout=config.dropout,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    metrics = _evaluate(model, data_loader)

    LOGGER.info("Test metrics: %s", metrics)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()