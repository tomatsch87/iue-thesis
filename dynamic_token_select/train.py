"""
Train sequence-aware iue model on token-level hidden-state sequences.

Default training behavior:
- Trains `dynamic_token_lstm` unless `--model` is changed.
- Loads `train`/`validation` splits from `--data-root` with layers `(48,)` by default.
- Uses `HiddenStateSequenceDataset` with `filter_empty=True` and no token filter unless provided.
- Runs a single W&B run by default and saves best checkpoint to `<output-dir>/best_model.pt`.

Important options:
- `--model`: `dynamic_token_lstm`, `abmil`, or `fixed_token_mlp`.
- Data/filtering: `--layers`, `--filter-empty`, `--token-filter`,
  `--token-filter-config` (JSON), `--tokenizer-name`.
- Optimization: `--batch-size`, `--max-steps`, `--patience`, `--lr`,
  `--weight-decay`, `--lr-scheduler`, `--warmup-steps`, `--min-lr`,
  `--gradient-clip-val`.
- Logging/sweeps: `--eval-every`, `--log-every`, `--run-sweep`,
  `--sweep-count`, `--sweep-method`, `--sweep-name`.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError as exc:
    raise ImportError("wandb must be installed to run dynamic token selection training") from exc

try:
    from sklearn.metrics import f1_score, roc_auc_score
except ImportError:
    roc_auc_score = None
    f1_score = None

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from dynamic_token_select.abmil import ABMILConfig, ABMILBagModel, ABMIL_SWEEP_CONFIG
from dynamic_token_select.dataset import HiddenStateSequenceDataset
from dynamic_token_select.lstm import (
    DYNAMIC_TOKEN_LSTM_SWEEP_CONFIG,
    DynamicTokenLSTM,
    LSTMConfig,
)
from dynamic_token_select.utils import DEVICE, bss, seed_everything
from dynamic_token_select.mlp import (
    FIXED_TOKEN_MLP_SWEEP_CONFIG,
    FixedTokenMLP,
    MLPConfig,
)

LOGGER = logging.getLogger(__name__)


class ModelSpec:
    def __init__(
        self,
        name: str,
        config_cls: Type[Any],
        model_builder: Callable[[int, Any], nn.Module],
        sweep_config: Mapping[str, Any],
    ) -> None:
        self.name = name
        self.config_cls = config_cls
        self.model_builder = model_builder
        self._sweep_config = dict(sweep_config)

    def sweep_config(self, method_override: Optional[str] = None) -> Dict[str, Any]:
        config_copy = copy.deepcopy(self._sweep_config)
        if method_override is not None:
            config_copy["method"] = method_override
        return config_copy

    @property
    def default_sweep_method(self) -> str:
        return str(self._sweep_config.get("method", "bayes"))


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "dynamic_token_lstm": ModelSpec(
        name="dynamic_token_lstm",
        config_cls=LSTMConfig,
        model_builder=lambda input_dim, cfg: DynamicTokenLSTM(
            input_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout,
        ),
        sweep_config=DYNAMIC_TOKEN_LSTM_SWEEP_CONFIG,
    ),
    "abmil": ModelSpec(
        name="abmil",
        config_cls=ABMILConfig,
        model_builder=lambda _input_dim, cfg: ABMILBagModel(cfg),
        sweep_config=ABMIL_SWEEP_CONFIG,
    ),
    "fixed_token_mlp": ModelSpec(
        name="fixed_token_mlp",
        config_cls=MLPConfig,
        model_builder=lambda input_dim, cfg: FixedTokenMLP(
            per_token_dim=input_dim,
            k=cfg.k,
            mlp_layers=cfg.mlp_layers,
            dropout=cfg.dropout,
        ),
        sweep_config=FIXED_TOKEN_MLP_SWEEP_CONFIG,
    ),
}

DEFAULT_MODEL = "dynamic_token_lstm"


def _json_object(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON for token filter config: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Token filter config must be a JSON object.")
    return parsed


def _make_sequence_collate(expected_length: Optional[int] = None) -> Callable[[Sequence[Tuple[torch.Tensor, torch.Tensor]]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    def _collate(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences, labels = zip(*batch)
        flattened = [seq.reshape(seq.size(0), -1).to(torch.float32) for seq in sequences]
        lengths = torch.tensor([seq.size(0) for seq in flattened], dtype=torch.long)
        if expected_length is not None and not torch.all(lengths == expected_length):
            raise ValueError(
                f"All sequences must have length {expected_length} for this model, got lengths={lengths.tolist()}"
            )
        padded = pad_sequence(flattened, batch_first=True, padding_value=0.0)
        stacked_labels = torch.stack(labels).to(torch.float32)
        return padded, lengths, stacked_labels

    return _collate


def create_dataloaders(config: Any) -> Tuple[DataLoader, DataLoader]:
    token_filter_config: Optional[Dict[str, Any]] = None
    if getattr(config, "token_filter_config", None) is not None:
        token_filter_config = dict(config.token_filter_config)
    if getattr(config, "token_filter", None):
        filter_name = str(config.token_filter)
        if "top_k-token-entropy" in filter_name and getattr(config, "k", None) is not None:
            if token_filter_config is None:
                token_filter_config = {}
            token_filter_config["k"] = config.k

    common_kwargs = dict(
        root=config.data_root,
        layers=config.layers,
        filter_empty=config.filter_empty,
        token_filter=config.token_filter,
        token_filter_config=token_filter_config,
        tokenizer_name=config.tokenizer_name,
    )
    train_dataset = HiddenStateSequenceDataset(split=config.train_split, **common_kwargs)
    valid_dataset = HiddenStateSequenceDataset(split=config.valid_split, **common_kwargs)

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=_make_sequence_collate(getattr(config, "k", None)),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_kwargs)
    valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, valid_loader


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    all_labels: List[float] = []
    all_probs: List[float] = []
    all_preds: List[float] = []
    correct = 0
    total = 0

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
        metrics["bss"] = bss(probs_array, labels_array)
    else:
        metrics["bss"] = float("nan")

    roc_auc = _compute_roc_auc(all_labels, all_probs)
    f1 = _compute_f1(all_labels, all_preds)
    metrics["roc_auc"] = roc_auc
    metrics["f1"] = f1
    return metrics


def _build_scheduler(optimizer: torch.optim.Optimizer, config: Any) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    scheduler_name = str(getattr(config, "lr_scheduler", "none")).lower()
    if scheduler_name != "cosine_warmup":
        return None

    total_steps = int(getattr(config, "max_steps", 0))
    if total_steps <= 0:
        return None

    warmup_steps = max(0, int(getattr(config, "warmup_steps", 0)))
    warmup_steps = min(warmup_steps, total_steps)
    min_lr = max(0.0, float(getattr(config, "min_lr", 0.0)))
    base_lr = float(getattr(config, "lr", 0.0))
    if base_lr <= 0:
        return None

    min_factor = min(min_lr / base_lr, 1.0)

    def lr_lambda(step: int) -> float:
        step = min(step, total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            warmup_progress = (step + 1) / warmup_steps
            return min_factor + (1.0 - min_factor) * warmup_progress

        remaining = max(total_steps - warmup_steps, 1)
        progress = (step - warmup_steps) / remaining
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.step()  # first batch uses warmup-scaled lr
    return scheduler


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


def train_once(spec: ModelSpec, base_config: Any) -> None:
    seed_everything(base_config.seed)
    base_config.output_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=base_config.wandb_project,
        name=base_config.run_name,
        config=base_config.to_logging_dict(),
    )

    config = spec.config_cls.from_wandb_config(base_config, dict(run.config))
    LOGGER.info("Resolved configuration for model '%s': %s", spec.name, config)

    train_loader, valid_loader = create_dataloaders(config)

    model = spec.model_builder(config.input_dim, config).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = _build_scheduler(optimizer, config)
    criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    best_metric = float("-inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_checkpoint_path = config.output_dir / "best_model.pt"
    patience_limit = getattr(config, "patience", None)
    patience_counter = 0
    early_stop_triggered = False
    first_epoch_logged = False

    while global_step < config.max_steps and not early_stop_triggered:
        epoch_start = time.perf_counter()
        model.train()
        for inputs, lengths, labels in train_loader:
            global_step += 1
            inputs = inputs.to(DEVICE, non_blocking=True)
            lengths = lengths.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            if hasattr(config, "gradient_clip_val") and config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if global_step % config.log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log({"train/loss": loss.item(), "train/lr": current_lr, "step": global_step})

            if global_step % config.eval_every == 0 or global_step >= config.max_steps:
                val_metrics = evaluate(model, valid_loader, criterion)
                wandb.log(
                    {
                        "val/loss": val_metrics["loss"],
                        "val/accuracy": val_metrics["accuracy"],
                        "val/bss": val_metrics["bss"],
                        "val/roc_auc": val_metrics["roc_auc"],
                        "val/f1": val_metrics["f1"],
                        "step": global_step,
                    }
                )
                model.train()
                target_metric = val_metrics.get("roc_auc")
                if target_metric is None or target_metric != target_metric:
                    target_metric = val_metrics.get("accuracy", float("-inf"))
                if target_metric > best_metric:
                    best_metric = target_metric
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_limit is not None and patience_limit > 0 and patience_counter >= patience_limit:
                        early_stop_triggered = True
                        LOGGER.info("Early stopping triggered at step %s", global_step)
                        break

            if global_step >= config.max_steps:
                break

        if not first_epoch_logged:
            first_epoch_logged = True
            epoch_duration = time.perf_counter() - epoch_start
            LOGGER.info("First epoch completed in %.2f seconds", epoch_duration)

    if best_state is not None:
        torch.save({"state_dict": best_state, "config": config.to_logging_dict()}, best_checkpoint_path)
        wandb.log({"best_metric": best_metric})
        wandb.summary["best_metric"] = best_metric
        wandb.summary["best_model_path"] = str(best_checkpoint_path)
        LOGGER.info("Saved best model for '%s' to %s", spec.name, best_checkpoint_path)

    wandb.summary["early_stopped"] = early_stop_triggered
    if early_stop_triggered:
        wandb.summary["early_stop_step"] = global_step

    run.finish()


def launch_sweep(spec: ModelSpec, base_config: Any, sweep_count: int, sweep_method: str) -> None:
    sweep_config = spec.sweep_config(method_override=sweep_method)
    sweep_config["name"] = base_config.sweep_name or f"{spec.name}_sweep"

    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=base_config.wandb_project,
    )
    LOGGER.info("Created W&B sweep %s for model '%s'", sweep_id, spec.name)

    wandb.agent(sweep_id, function=lambda: train_once(spec, base_config), count=sweep_count)


def _prepare_config_kwargs(args: argparse.Namespace, spec: ModelSpec) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    field_names = {field.name for field in fields(spec.config_cls)}

    for name in field_names:
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None:
            continue
        if name == "layers":
            kwargs[name] = tuple(sorted(set(value))) if value else None
        elif name in {"data_root", "output_dir"}:
            kwargs[name] = value.resolve()
        elif name == "token_filter_config":
            kwargs[name] = dict(value)
        else:
            kwargs[name] = value
    return kwargs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY.keys()),
        default=DEFAULT_MODEL,
        help="Model architecture to train",
    )
    model_args, _ = model_parser.parse_known_args(argv)
    spec = MODEL_REGISTRY[model_args.model]
    default_config = spec.config_cls()

    parser = argparse.ArgumentParser(
        description="Train sequential classifiers on hidden-state sequences",
        parents=[model_parser],
    )

    parser.add_argument("--data-root", type=Path, default=default_config.data_root, help="Root directory with extracted features")
    parser.add_argument("--train-split", default=default_config.train_split, help="Dataset split used for training")
    parser.add_argument("--valid-split", default=default_config.valid_split, help="Dataset split used for validation")
    parser.add_argument("--layers", type=int, nargs="*", default=list(default_config.layers) if default_config.layers else None, help="Hidden-state layers to use (omit to infer from files)")
    parser.add_argument("--filter-empty", action="store_true", default=default_config.filter_empty, help="Filter samples with empty spans")
    parser.add_argument("--token-filter", default=default_config.token_filter, help="Optional token filter strategy")
    parser.add_argument("--token-filter-config", type=_json_object, default=None, help="JSON string with token filter parameters")
    parser.add_argument("--tokenizer-name", default=default_config.tokenizer_name, help="Tokenizer to load when filtering tokens")
    if hasattr(default_config, "k"):
        parser.add_argument("--k", type=int, default=default_config.k, help="Expected fixed sequence length (requires filters producing exactly K tokens)")
    if hasattr(default_config, "hidden_dim"):
        parser.add_argument("--hidden-dim", type=int, default=getattr(default_config, "hidden_dim"), help="LSTM hidden size")
    parser.add_argument("--input-dim", type=int, default=default_config.input_dim, help="Per-token feature dimension fed to the model")
    if hasattr(default_config, "lstm_layers"):
        parser.add_argument(
            "--lstm-layers",
            type=int,
            default=getattr(default_config, "lstm_layers"),
            help="Number of stacked LSTM layers",
        )
    if hasattr(default_config, "mlp_layers"):
        parser.add_argument(
            "--mlp-layers",
            type=int,
            default=getattr(default_config, "mlp_layers"),
            help="Number of hidden layers in the fixed-length MLP",
        )
    if hasattr(default_config, "dropout"):
        parser.add_argument(
            "--dropout",
            type=float,
            default=getattr(default_config, "dropout"),
            help="Recurrent dropout between LSTM layers and on top of LSTM outputs",
        )

    if hasattr(default_config, "encoder_layers"):
        parser.add_argument(
            "--encoder-layers",
            type=int,
            default=getattr(default_config, "encoder_layers"),
            help="Number of hidden layers in the ABMIL instance encoder MLP",
        )
    if hasattr(default_config, "encoder_hidden_dim"):
        parser.add_argument(
            "--encoder-hidden-dim",
            type=int,
            default=getattr(default_config, "encoder_hidden_dim"),
            help="Hidden dimension used inside the ABMIL instance encoder",
        )
    if hasattr(default_config, "encoder_output_dim"):
        parser.add_argument(
            "--encoder-output-dim",
            type=int,
            default=getattr(default_config, "encoder_output_dim"),
            help="Output dimension produced by the ABMIL instance encoder",
        )
    if hasattr(default_config, "encoder_dropout"):
        parser.add_argument(
            "--encoder-dropout",
            type=float,
            default=getattr(default_config, "encoder_dropout"),
            help="Dropout applied inside the ABMIL instance encoder",
        )
    if hasattr(default_config, "att_dim"):
        parser.add_argument(
            "--att-dim",
            type=int,
            default=getattr(default_config, "att_dim"),
            help="Attention dimension for the ABMIL pooling module",
        )
    if hasattr(default_config, "att_act"):
        parser.add_argument(
            "--att-act",
            default=getattr(default_config, "att_act"),
            help="Attention activation for ABMIL (tanh, relu, gelu)",
        )
    if hasattr(default_config, "gated_attention"):
        gated_group = parser.add_mutually_exclusive_group()
        gated_group.add_argument(
            "--gated-attention",
            dest="gated_attention",
            action="store_true",
            help="Enable gated attention in ABMIL",
        )
        gated_group.add_argument(
            "--no-gated-attention",
            dest="gated_attention",
            action="store_false",
            help="Disable gated attention in ABMIL",
        )
        parser.set_defaults(gated_attention=getattr(default_config, "gated_attention"))
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size, help="Training batch size")
    parser.add_argument("--max-steps", type=int, default=default_config.max_steps, help="Total gradient steps")
    parser.add_argument("--patience", type=int, default=default_config.patience, help="Validation patience before early stopping (<=0 disables)")
    parser.add_argument("--lr", type=float, default=default_config.lr, help="Optimizer learning rate")
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine_warmup"],
        default=getattr(default_config, "lr_scheduler", "none"),
        help="Learning rate scheduler to use",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=getattr(default_config, "warmup_steps", 0),
        help="Number of warmup steps for cosine scheduler",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=getattr(default_config, "min_lr", 0.0),
        help="Minimum LR target for cosine scheduler",
    )
    parser.add_argument("--weight-decay", type=float, default=default_config.weight_decay, help="Weight decay")
    parser.add_argument("--gradient-clip-val", type=float, default=getattr(default_config, "gradient_clip_val", None), help="Gradient clipping value")
    parser.add_argument("--eval-every", type=int, default=default_config.eval_every, help="Evaluation frequency in steps")
    parser.add_argument("--log-every", type=int, default=default_config.log_every, help="Logging frequency in steps")
    parser.add_argument("--num-workers", type=int, default=default_config.num_workers, help="Dataloader worker processes")
    parser.add_argument("--seed", type=int, default=default_config.seed, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=default_config.output_dir, help="Directory to store checkpoints")
    parser.add_argument("--wandb-project", default=default_config.wandb_project, help="Weights & Biases project name")
    parser.add_argument("--run-name", default=default_config.run_name, help="Optional W&B run name")
    parser.add_argument("--run-sweep", action="store_true", help="Launch a W&B hyperparameter sweep instead of a single run")
    parser.add_argument("--sweep-count", type=int, default=10, help="Number of sweep runs to execute")
    parser.add_argument("--sweep-method", default=spec.default_sweep_method, help="W&B sweep search method")
    parser.add_argument("--sweep-name", default=default_config.sweep_name, help="Optional W&B sweep name")

    args = parser.parse_args(argv)
    if args.token_filter_config is None and default_config.token_filter_config:
        args.token_filter_config = dict(default_config.token_filter_config)
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    spec = MODEL_REGISTRY[args.model]
    config_kwargs = _prepare_config_kwargs(args, spec)
    config = spec.config_cls(**config_kwargs)

    if args.run_sweep:
        launch_sweep(spec, config, sweep_count=args.sweep_count, sweep_method=args.sweep_method)
    else:
        train_once(spec, config)


if __name__ == "__main__":
    main()