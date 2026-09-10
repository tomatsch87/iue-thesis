"""
Train a line-sequence LSTM iue model from cached hidden states.

Default training behavior:
- Requires `--train-jsonl` and `--valid-jsonl` aligned with cached feature files.
- Uses layers `(48,)`, feature splits `train/validation/test`, and
	`drop_ws_comment=True` by default.
- Builds variable-length per-line sequences via `LineSequenceHiddenStateDataset`.
- `pos_weight='auto'` computes class weighting from train data.
- Saves best model by default to `outputs/token_level/line_lstm`.

Important options:
- Data mapping: `--layers`, `--feature-splits`, `--drop-ws-comment`,
	`--include-ws-comment-tokens`, `--match-mode`, `--strict-mapping`.
- Model/optimization: `--hidden-dim`, `--lstm-layers`, `--bidirectional`, `--dropout`,
	`--batch-size`, `--max-steps`, `--patience`, `--lr`, `--pos-weight`,
	`--weight-decay`, `--gradient-clip-val`, `--lr-scheduler`, `--warmup-steps`, `--min-lr`.
- Run control: `--run-sweep`, `--sweep-count`, `--sweep-method`, `--save-best-model`.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

try:
	import wandb
except ImportError as exc:
	raise ImportError("wandb must be installed to run this script") from exc

try:
	from sklearn.metrics import f1_score, roc_auc_score
except ImportError:
	f1_score = None
	roc_auc_score = None

from .lstm import LINE_LSTM_SWEEP_CONFIG, LineContextLSTM, LineLSTMConfig
from .sequence_dataset import LineSequenceHiddenStateDataset
from .utils import DEVICE, bss, seed_everything


LOGGER = logging.getLogger(__name__)


def _json_object(value: str) -> Dict[str, Any]:
	import json

	try:
		parsed = json.loads(value)
	except json.JSONDecodeError as exc:
		raise argparse.ArgumentTypeError(f"Invalid JSON for token filter config: {exc}") from exc
	if not isinstance(parsed, dict):
		raise argparse.ArgumentTypeError("Expected a JSON object.")
	return parsed


def _parse_pos_weight_arg(value: str) -> Optional[float | str]:
	raw = str(value).strip().lower()
	if raw in {"none", "off", "disable", "disabled"}:
		return None
	if raw == "auto":
		return "auto"
	try:
		parsed = float(raw)
	except ValueError as exc:
		raise argparse.ArgumentTypeError("--pos-weight must be a float, 'auto', or 'none'") from exc
	if parsed <= 0.0:
		raise argparse.ArgumentTypeError("--pos-weight must be > 0 when set explicitly")
	return parsed


def _make_sequence_collate(
	batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	sequences, labels, _metadata = zip(*batch)
	lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
	padded_sequences = pad_sequence(list(sequences), batch_first=True, padding_value=0.0).to(torch.float32)
	stacked_labels = torch.stack(list(labels)).to(torch.float32)
	return padded_sequences, lengths, stacked_labels


def _prepare_dataset(
	root: Path,
	jsonl_path: Path,
	layers: Optional[Sequence[int]],
	feature_splits: Sequence[str],
	drop_ws_comment: bool,
	include_ws_comment_tokens: bool,
	match_mode: str,
	strict_mapping: bool,
) -> LineSequenceHiddenStateDataset:
	return LineSequenceHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=layers,
		feature_splits=feature_splits,
		drop_ws_comment=drop_ws_comment,
		include_ws_comment_tokens=include_ws_comment_tokens,
		match_mode=match_mode,
		strict_mapping=strict_mapping,
	)


def create_dataloaders(config: LineLSTMConfig) -> Tuple[DataLoader, DataLoader, LineSequenceHiddenStateDataset, LineSequenceHiddenStateDataset]:
	if config.train_jsonl is None or config.valid_jsonl is None:
		raise ValueError("Both train_jsonl and valid_jsonl must be provided")

	train_dataset = _prepare_dataset(
		root=config.data_root,
		jsonl_path=config.train_jsonl,
		layers=config.layers,
		feature_splits=config.feature_splits,
		drop_ws_comment=config.drop_ws_comment,
		include_ws_comment_tokens=config.include_ws_comment_tokens,
		match_mode=config.match_mode,
		strict_mapping=config.strict_mapping,
	)
	valid_dataset = _prepare_dataset(
		root=config.data_root,
		jsonl_path=config.valid_jsonl,
		layers=train_dataset.layers,
		feature_splits=config.feature_splits,
		drop_ws_comment=config.drop_ws_comment,
		include_ws_comment_tokens=config.include_ws_comment_tokens,
		match_mode=config.match_mode,
		strict_mapping=config.strict_mapping,
	)

	train_loader = DataLoader(
		train_dataset,
		batch_size=config.batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=config.num_workers,
		pin_memory=(DEVICE.type == "cuda"),
		collate_fn=_make_sequence_collate,
	)
	valid_loader = DataLoader(
		valid_dataset,
		batch_size=config.batch_size,
		shuffle=False,
		drop_last=False,
		num_workers=config.num_workers,
		pin_memory=(DEVICE.type == "cuda"),
		collate_fn=_make_sequence_collate,
	)
	return train_loader, valid_loader, train_dataset, valid_dataset


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
	metrics["roc_auc"] = _compute_roc_auc(all_labels, all_probs)
	metrics["f1"] = _compute_f1(all_labels, all_preds)
	return metrics


def _build_scheduler(optimizer: torch.optim.Optimizer, config: LineLSTMConfig) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
	scheduler_name = str(config.lr_scheduler).lower()
	if scheduler_name != "cosine_warmup":
		return None

	total_steps = int(config.max_steps)
	if total_steps <= 0:
		return None

	warmup_steps = max(0, int(config.warmup_steps))
	warmup_steps = min(warmup_steps, total_steps)
	min_lr = max(0.0, float(config.min_lr))
	base_lr = float(config.lr)
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
	scheduler.step()
	return scheduler


def _build_model(config: LineLSTMConfig) -> LineContextLSTM:
	return LineContextLSTM(
		input_dim=config.input_dim,
		hidden_dim=config.hidden_dim,
		num_layers=config.lstm_layers,
		dropout=config.dropout,
		bidirectional=config.bidirectional,
	)


def _sanitize_path_component(value: str) -> str:
	text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
	text = text.strip("._-")
	return text or "run"


def _resolve_best_checkpoint_path(config: LineLSTMConfig, run: Any) -> Path:
	base_dir = config.output_dir
	if not bool(config.save_best_per_run):
		return base_dir / "best_line_lstm.pt"

	run_id = str(getattr(run, "id", "") or "run")
	run_name = str(getattr(run, "name", "") or "run")
	run_dir_name = f"{_sanitize_path_component(run_name)}_{_sanitize_path_component(run_id)}"
	run_dir = base_dir / run_dir_name
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir / "best_line_lstm.pt"


def _resolve_pos_weight(config: LineLSTMConfig, train_dataset: LineSequenceHiddenStateDataset) -> Optional[float]:
	configured = config.pos_weight
	if configured is None:
		LOGGER.info("Using unweighted BCEWithLogitsLoss (pos_weight disabled)")
		return None

	if isinstance(configured, str) and configured.strip().lower() == "auto":
		pos_count = 0
		neg_count = 0
		for item in train_dataset._items:  # pylint: disable=protected-access
			if float(item.label) >= 0.5:
				pos_count += 1
			else:
				neg_count += 1
		if pos_count <= 0 or neg_count <= 0:
			LOGGER.warning(
				"Could not auto-compute pos_weight due to single-class train data (pos=%d, neg=%d); defaulting to 1.0",
				pos_count,
				neg_count,
			)
			return 1.0
		resolved = float(neg_count / pos_count)
		LOGGER.info("Auto-computed pos_weight=%.6f (neg=%d, pos=%d)", resolved, neg_count, pos_count)
		return resolved

	resolved = float(configured)
	if resolved <= 0.0:
		raise ValueError("pos_weight must be > 0 when set explicitly")
	LOGGER.info("Using explicit pos_weight=%.6f", resolved)
	return resolved


def train_once(base_config: LineLSTMConfig) -> None:
	seed_everything(base_config.seed)
	base_config.output_dir.mkdir(parents=True, exist_ok=True)

	run = wandb.init(
		project=base_config.wandb_project,
		name=base_config.run_name,
		config=base_config.to_logging_dict(),
	)
	config = LineLSTMConfig.from_wandb_config(base_config, dict(run.config))
	if config.train_jsonl is None or config.valid_jsonl is None:
		raise ValueError("train_jsonl and valid_jsonl are required")

	train_loader, valid_loader, train_dataset, valid_dataset = create_dataloaders(config)

	if train_dataset.feature_dim is None:
		raise RuntimeError("Unable to infer feature dimension for train dataset")
	config.input_dim = int(train_dataset.feature_dim)
	if train_dataset.layers is not None:
		config.layers = tuple(train_dataset.layers)

	model = _build_model(config).to(DEVICE)
	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	scheduler = _build_scheduler(optimizer, config)
	resolved_pos_weight = _resolve_pos_weight(config, train_dataset)
	if resolved_pos_weight is None:
		criterion = nn.BCEWithLogitsLoss()
	else:
		criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(resolved_pos_weight, device=DEVICE))

	wandb.config.update(
		{
			"input_dim": config.input_dim,
			"layers": list(config.layers or []),
			"resolved_pos_weight": resolved_pos_weight,
		},
		allow_val_change=True,
	)
	LOGGER.info("Resolved configuration: %s", config)
	LOGGER.info("train sequences=%d feature_dim=%s stats=%s", len(train_dataset), train_dataset.feature_dim, train_dataset.stats)
	LOGGER.info("valid sequences=%d feature_dim=%s stats=%s", len(valid_dataset), valid_dataset.feature_dim, valid_dataset.stats)

	global_step = 0
	best_metric = float("-inf")
	best_state: Optional[Dict[str, torch.Tensor]] = None
	best_checkpoint_path = _resolve_best_checkpoint_path(config, run)
	patience_limit = config.patience
	patience_counter = 0
	early_stop_triggered = False

	while global_step < config.max_steps and not early_stop_triggered:
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
			if config.gradient_clip_val is not None:
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
			optimizer.step()
			if scheduler is not None:
				scheduler.step()

			if global_step % config.log_every == 0:
				with torch.no_grad():
					train_probs = torch.sigmoid(logits)
					train_preds = (train_probs >= 0.5).float()
					train_labels = labels
					train_acc = float((train_preds == train_labels).float().mean().item()) if train_labels.numel() else 0.0
				current_lr = float(optimizer.param_groups[0]["lr"])
				wandb.log({"train/loss": float(loss.item()), "train/accuracy": train_acc, "train/lr": current_lr, "step": global_step})

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

	if bool(config.save_best_model) and best_state is not None:
		checkpoint = {
			"state_dict": best_state,
			"config": config.to_logging_dict(),
			"run_id": str(getattr(run, "id", "") or ""),
			"run_name": str(getattr(run, "name", "") or ""),
		}
		torch.save(checkpoint, best_checkpoint_path)
		wandb.log({"best_metric": best_metric})
		wandb.summary["best_metric"] = best_metric
		wandb.summary["best_model_path"] = str(best_checkpoint_path)
		LOGGER.info("Saved best model to %s", best_checkpoint_path)

	wandb.summary["early_stopped"] = early_stop_triggered
	if early_stop_triggered:
		wandb.summary["early_stop_step"] = global_step
	run.finish()


def launch_sweep(base_config: LineLSTMConfig, sweep_count: int, sweep_method: str) -> None:
	sweep_config = copy.deepcopy(LINE_LSTM_SWEEP_CONFIG)
	sweep_config["method"] = sweep_method
	sweep_config["name"] = base_config.sweep_name or "line_lstm_sweep"

	sweep_id = wandb.sweep(
		sweep=sweep_config,
		project=base_config.wandb_project,
	)
	LOGGER.info("Created W&B sweep %s", sweep_id)
	wandb.agent(sweep_id, function=lambda: train_once(base_config), count=sweep_count)


def _prepare_config_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
	kwargs: Dict[str, Any] = {}
	field_names = {field.name for field in fields(LineLSTMConfig)}
	for name in field_names:
		if not hasattr(args, name):
			continue
		value = getattr(args, name)
		if value is None:
			continue
		if name == "layers":
			kwargs[name] = tuple(sorted(set(int(item) for item in value))) if value else None
		elif name in {"data_root", "output_dir", "train_jsonl", "valid_jsonl"}:
			kwargs[name] = Path(value).resolve()
		elif name == "feature_splits":
			kwargs[name] = tuple(str(item) for item in value)
		else:
			kwargs[name] = value
	return kwargs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	default_config = LineLSTMConfig()
	parser = argparse.ArgumentParser(
		description="Train a line-sequence LSTM hallucination detector on cached hidden-state features",
	)
	parser.add_argument("--data-root", type=Path, default=default_config.data_root)
	parser.add_argument("--train-jsonl", type=Path, required=True)
	parser.add_argument("--valid-jsonl", type=Path, required=True)
	parser.add_argument("--layers", type=int, nargs="*", default=list(default_config.layers) if default_config.layers else None)
	parser.add_argument("--feature-splits", nargs="*", default=tuple(default_config.feature_splits))
	parser.add_argument("--drop-ws-comment", action="store_true", default=default_config.drop_ws_comment)
	parser.add_argument("--include-ws-comment-tokens", action="store_true", default=default_config.include_ws_comment_tokens)
	parser.add_argument("--match-mode", choices=["auto", "index", "fingerprint"], default=default_config.match_mode)
	parser.add_argument("--strict-mapping", action="store_true", default=default_config.strict_mapping)
	parser.add_argument("--hidden-dim", type=int, default=default_config.hidden_dim)
	parser.add_argument("--lstm-layers", type=int, default=default_config.lstm_layers)
	parser.add_argument("--bidirectional", action="store_true", default=default_config.bidirectional)
	parser.add_argument("--dropout", type=float, default=default_config.dropout)
	parser.add_argument("--batch-size", type=int, default=default_config.batch_size)
	parser.add_argument("--max-steps", type=int, default=default_config.max_steps)
	parser.add_argument("--patience", type=int, default=default_config.patience)
	parser.add_argument("--lr", type=float, default=default_config.lr)
	parser.add_argument("--pos-weight", type=_parse_pos_weight_arg, default=default_config.pos_weight)
	parser.add_argument("--weight-decay", type=float, default=default_config.weight_decay)
	parser.add_argument("--gradient-clip-val", type=float, default=default_config.gradient_clip_val)
	parser.add_argument("--lr-scheduler", choices=["none", "cosine_warmup"], default=default_config.lr_scheduler)
	parser.add_argument("--warmup-steps", type=int, default=default_config.warmup_steps)
	parser.add_argument("--min-lr", type=float, default=default_config.min_lr)
	parser.add_argument("--eval-every", type=int, default=default_config.eval_every)
	parser.add_argument("--log-every", type=int, default=default_config.log_every)
	parser.add_argument("--num-workers", type=int, default=default_config.num_workers)
	parser.add_argument("--seed", type=int, default=default_config.seed)
	parser.add_argument("--output-dir", type=Path, default=default_config.output_dir)
	parser.add_argument("--save-best-model", action=argparse.BooleanOptionalAction, default=default_config.save_best_model)
	parser.add_argument("--save-best-per-run", action=argparse.BooleanOptionalAction, default=None)
	parser.add_argument("--wandb-project", default=default_config.wandb_project)
	parser.add_argument("--run-name", default=default_config.run_name)
	parser.add_argument("--run-sweep", action="store_true")
	parser.add_argument("--sweep-count", type=int, default=10)
	parser.add_argument("--sweep-method", default=LINE_LSTM_SWEEP_CONFIG.get("method", "bayes"))
	parser.add_argument("--sweep-name", default=default_config.sweep_name)
	parser.add_argument("--_unused", type=_json_object, default=None, help=argparse.SUPPRESS)
	return parser.parse_args(argv)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	args = parse_args()
	config_kwargs = _prepare_config_kwargs(args)
	if args.save_best_per_run is None:
		config_kwargs["save_best_per_run"] = bool(args.run_sweep)
	config = LineLSTMConfig(**config_kwargs)

	if config.train_jsonl is None or config.valid_jsonl is None:
		raise ValueError("Both --train-jsonl and --valid-jsonl are required")

	if args.run_sweep:
		launch_sweep(config, sweep_count=args.sweep_count, sweep_method=args.sweep_method)
	else:
		train_once(config)


if __name__ == "__main__":
	main()