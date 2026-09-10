"""
Train baseline MLP iue model on cached hidden-state features.

Default behavior:
- Trains the `snyder` baseline unless `--baseline` is set to `openia`.
- Loads `train` and `validation` splits from `--data-root`.
- Uses `HiddenStateDataset` with combiner `concat`, no empty-span filtering by default.
- Runs a single W&B run (not a sweep) and saves best checkpoint to
	`<output-dir>/best_model.pt`.
- Evaluates every `eval_every` steps and early-stops only when `patience > 0`.

Important options:
- `--baseline`: `snyder` or `openia` architecture/config defaults.
- `--layers`, `--token-positions`, `--combiner`, `--filter-empty`: feature selection.
- `--batch-size`, `--max-steps`, `--lr`, `--weight-decay`, `--patience`: training control.
- `--run-sweep`, `--sweep-count`, `--sweep-method`, `--sweep-name`: launch W&B sweeps.
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
	import wandb
except ImportError as exc:
	raise ImportError("wandb must be installed to run baseline training") from exc

try:
	from sklearn.metrics import f1_score, roc_auc_score
except ImportError:
	roc_auc_score = None
	f1_score = None

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
	sys.path.insert(0, str(PACKAGE_PARENT))

from baselines.dataset import HiddenStateDataset
from baselines.snyder import SNYDER_SWEEP_CONFIG, SnyderConfig, SnyderMLP
from baselines.openia import OPENIA_SWEEP_CONFIG, OpeniaConfig, OpeniaMLP
from baselines.utils import DEVICE, seed_everything

LOGGER = logging.getLogger(__name__)


class BaselineSpec:
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


BASELINE_REGISTRY: Dict[str, BaselineSpec] = {
	"snyder": BaselineSpec(
		name="snyder",
		config_cls=SnyderConfig,
		model_builder=lambda input_dim, cfg: SnyderMLP(input_dim=input_dim, hidden_dim=cfg.hidden_dim),
		sweep_config=SNYDER_SWEEP_CONFIG,
	),
	"openia": BaselineSpec(
		name="openia",
		config_cls=OpeniaConfig,
		model_builder=lambda input_dim, cfg: OpeniaMLP(input_dim=input_dim, hidden_dim=cfg.hidden_dim),
		sweep_config=OPENIA_SWEEP_CONFIG,
	),
}

DEFAULT_BASELINE = "snyder"


def create_dataloaders(config: Any) -> Tuple[DataLoader, DataLoader, int]:
	train_dataset = HiddenStateDataset(
		root=config.data_root,
		split=config.train_split,
		layers=config.layers,
		token_positions=config.token_positions,
		filter_empty=config.filter_empty,
		combiner=config.combiner,
	)
	valid_dataset = HiddenStateDataset(
		root=config.data_root,
		split=config.valid_split,
		layers=config.layers,
		token_positions=config.token_positions,
		filter_empty=config.filter_empty,
		combiner=config.combiner,
	)

	input_dim = train_dataset.feature_dim
	if input_dim is None:
		raise RuntimeError("Training dataset did not expose a feature dimension")

	train_loader = DataLoader(
		train_dataset,
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=config.num_workers,
		pin_memory=(DEVICE.type == "cuda"),
		drop_last=True,
	)
	valid_loader = DataLoader(
		valid_dataset,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.num_workers,
		pin_memory=(DEVICE.type == "cuda"),
	)
	return train_loader, valid_loader, input_dim


def evaluate(model: nn.Module, data_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
	model.eval()
	losses: List[float] = []
	all_labels: List[float] = []
	all_probs: List[float] = []
	all_preds: List[float] = []
	correct = 0
	total = 0

	with torch.no_grad():
		for features, labels in data_loader:
			features = features.to(DEVICE)
			labels = labels.to(DEVICE)
			logits = model(features)
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

	if roc_auc_score is not None:
		try:
			metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs))
		except ValueError:
			metrics["roc_auc"] = float("nan")
	else:
		metrics["roc_auc"] = float("nan")

	if f1_score is not None:
		try:
			metrics["f1"] = float(f1_score(all_labels, all_preds))
		except ValueError:
			metrics["f1"] = float("nan")
	else:
		metrics["f1"] = float("nan")

	return metrics


def train_once(spec: BaselineSpec, base_config: Any) -> None:
	seed_everything(base_config.seed)
	base_config.output_dir.mkdir(parents=True, exist_ok=True)

	run = wandb.init(
		project=base_config.wandb_project,
		name=base_config.run_name,
		config=base_config.to_logging_dict(),
	)

	config = spec.config_cls.from_wandb_config(base_config, dict(run.config))
	LOGGER.info("Resolved configuration for baseline '%s': %s", spec.name, config)

	train_loader, valid_loader, input_dim = create_dataloaders(config)
	LOGGER.info("Input feature dimension: %s", input_dim)

	model = spec.model_builder(input_dim, config)
	model.to(DEVICE)

	optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	criterion = nn.BCEWithLogitsLoss()

	global_step = 0
	best_metric = float("-inf")
	best_state: Optional[Dict[str, torch.Tensor]] = None
	best_checkpoint_path = config.output_dir / "best_model.pt"
	patience_limit = getattr(config, "patience", None)
	patience_counter = 0
	early_stop_triggered = False

	while global_step < config.max_steps and not early_stop_triggered:
		model.train()
		for features, labels in train_loader:
			global_step += 1
			features = features.to(DEVICE)
			labels = labels.to(DEVICE)

			optimizer.zero_grad(set_to_none=True)
			logits = model(features)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

			if global_step % config.log_every == 0:
				wandb.log({"train/loss": loss.item(), "step": global_step})

			if global_step % config.eval_every == 0 or global_step == config.max_steps:
				val_metrics = evaluate(model, valid_loader, criterion)
				wandb.log({
					"val/loss": val_metrics["loss"],
					"val/accuracy": val_metrics["accuracy"],
					"val/roc_auc": val_metrics["roc_auc"],
					"val/f1": val_metrics["f1"],
					"step": global_step,
				})
				model.train()
				target_metric = val_metrics.get("roc_auc")
				if target_metric is None or target_metric != target_metric:  # NaN guard
					target_metric = val_metrics["f1"]
				improved = target_metric > best_metric
				if improved:
					best_metric = target_metric
					best_state = {k: v.cpu() for k, v in model.state_dict().items()}
					patience_counter = 0
				else:
					patience_counter += 1
					if patience_limit is not None and patience_limit > 0 and patience_counter >= patience_limit:
						LOGGER.info(
							"Early stopping triggered after %d validation checks without improvement",
							patience_limit,
						)
						wandb.log({"training/early_stopped": True, "step": global_step})
						early_stop_triggered = True
						break

			if global_step >= config.max_steps or early_stop_triggered:
				break

	if best_state is not None:
		torch.save({"state_dict": best_state, "config": config.to_logging_dict()}, best_checkpoint_path)
		wandb.log({"best_metric": best_metric})
		wandb.summary["best_metric"] = best_metric
		wandb.summary["best_model_path"] = str(best_checkpoint_path)
		LOGGER.info("Saved best model for baseline '%s' to %s", spec.name, best_checkpoint_path)

	wandb.summary["early_stopped"] = early_stop_triggered
	if early_stop_triggered:
		wandb.summary["early_stop_step"] = global_step

	run.finish()


def launch_sweep(spec: BaselineSpec, base_config: Any, sweep_count: int, sweep_method: str) -> None:
	sweep_config = spec.sweep_config(method_override=sweep_method)
	sweep_config["name"] = base_config.sweep_name or f"{spec.name}_sweep"
	
	sweep_id = wandb.sweep(
		sweep=sweep_config,
		project=base_config.wandb_project,
	)
	LOGGER.info("Created W&B sweep %s for baseline '%s'", sweep_id, spec.name)

	wandb.agent(sweep_id, function=lambda: train_once(spec, base_config), count=sweep_count)


def _prepare_config_kwargs(args: argparse.Namespace, spec: BaselineSpec) -> Dict[str, Any]:
	kwargs: Dict[str, Any] = {}
	field_names = {field.name for field in fields(spec.config_cls)}

	for name in field_names:
		if not hasattr(args, name):
			continue
		value = getattr(args, name)
		if name in {"layers", "token_positions"} and value is not None:
			kwargs[name] = tuple(sorted(set(value)))
		elif name in {"data_root", "output_dir"} and value is not None:
			kwargs[name] = value.resolve()
		else:
			kwargs[name] = value

	return kwargs


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	baseline_parser = argparse.ArgumentParser(add_help=False)
	baseline_parser.add_argument(
		"--baseline",
		choices=sorted(BASELINE_REGISTRY.keys()),
		default=DEFAULT_BASELINE,
		help="Baseline architecture to train",
	)
	baseline_args, _ = baseline_parser.parse_known_args(argv)
	spec = BASELINE_REGISTRY[baseline_args.baseline]
	default_config = spec.config_cls()

	parser = argparse.ArgumentParser(
		description="Train baseline MLP classifiers on extracted hidden states",
		parents=[baseline_parser],
	)
	parser.add_argument("--data-root", type=Path, default=default_config.data_root, help="Root directory with extracted features")
	parser.add_argument("--train-split", default=default_config.train_split, help="Dataset split used for training")
	parser.add_argument("--valid-split", default=default_config.valid_split, help="Dataset split used for validation")
	parser.add_argument("--layers", type=int, nargs="*", default=list(default_config.layers), help="Hidden-state layers to use")
	parser.add_argument("--token-positions", type=int, nargs="*", default=list(default_config.token_positions), help="Relative assistant token positions to use")
	parser.add_argument("--filter-empty", action="store_true", default=default_config.filter_empty, help="Filter samples whose code span is empty")
	parser.add_argument("--combiner", choices=["concat", "average"], default=default_config.combiner, help="How to combine selected hidden-state vectors")
	parser.add_argument("--hidden-dim", type=int, default=default_config.hidden_dim, help="Hidden layer width")
	parser.add_argument("--batch-size", type=int, default=default_config.batch_size, help="Training batch size")
	parser.add_argument("--max-steps", type=int, default=default_config.max_steps, help="Total gradient steps")
	parser.add_argument("--patience", type=int, default=default_config.patience, help="Number of validation checks to wait for improvement before early stopping (set <=0 to disable)")
	parser.add_argument("--lr", type=float, default=default_config.lr, help="Adam learning rate")
	parser.add_argument("--weight-decay", type=float, default=default_config.weight_decay, help="Adam weight decay")
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
	return parser.parse_args(argv)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
	args = parse_args()

	spec = BASELINE_REGISTRY[args.baseline]
	config_kwargs = _prepare_config_kwargs(args, spec)
	config = spec.config_cls(**config_kwargs)

	if args.run_sweep:
		launch_sweep(spec, config, sweep_count=args.sweep_count, sweep_method=args.sweep_method)
	else:
		train_once(spec, config)


if __name__ == "__main__":
	main()