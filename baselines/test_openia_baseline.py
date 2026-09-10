from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional
import sys

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
	sys.path.insert(0, str(PACKAGE_PARENT))
	
from baselines.baseline import evaluate
from baselines.dataset import HiddenStateDataset
from baselines.openia import OpeniaConfig, OpeniaMLP
from baselines.utils import DEVICE

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate a trained OpenIA baseline model on the test split")
	parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to the saved checkpoint (.pt)")
	parser.add_argument("--test-split", default="test", help="Dataset split to evaluate")
	parser.add_argument("--data-root", type=Path, help="Optional override for the data root; defaults to checkpoint config")
	parser.add_argument("--batch-size", type=int, help="Optional override for evaluation batch size")
	parser.add_argument("--num-workers", type=int, default=0, help="Dataloader worker processes")
	parser.add_argument("--limit-test", type=int, help="Optional cap on the number of test samples to load")
	return parser.parse_args()


def _load_checkpoint(path: Path) -> tuple[dict, dict]:
	checkpoint = torch.load(path, map_location="cpu")
	if not isinstance(checkpoint, dict):
		raise RuntimeError("Checkpoint should be a dict with 'state_dict' and 'config'")
	if "state_dict" not in checkpoint or "config" not in checkpoint:
		raise RuntimeError("Checkpoint missing required keys 'state_dict' and/or 'config'")
	return checkpoint["state_dict"], checkpoint["config"]


def _prepare_config(saved_config: dict, overrides: argparse.Namespace) -> OpeniaConfig:
	base_cfg = OpeniaConfig()
	cfg = OpeniaConfig.from_wandb_config(base_cfg, saved_config)
	if overrides.data_root is not None:
		cfg.data_root = overrides.data_root.resolve()
	if overrides.batch_size is not None:
		cfg.batch_size = int(overrides.batch_size)
	if overrides.num_workers is not None:
		cfg.num_workers = int(overrides.num_workers)
	return cfg


def _dataset_from_config(config: OpeniaConfig, split: str, limit: Optional[int]) -> HiddenStateDataset:
	dataset = HiddenStateDataset(
		root=config.data_root,
		split=split,
		layers=config.layers,
		token_positions=config.token_positions,
		filter_empty=config.filter_empty,
		combiner=config.combiner,
	)
	if limit is not None and limit < len(dataset):
		indices = list(range(limit))
		dataset = torch.utils.data.Subset(dataset, indices)  # type: ignore[arg-type]
	return dataset # type: ignore


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	args = _parse_args()

	state_dict, saved_config = _load_checkpoint(args.checkpoint_path)
	config = _prepare_config(saved_config, args)

	LOGGER.info("Evaluating checkpoint %s on split '%s'", args.checkpoint_path, args.test_split)
	LOGGER.info("Resolved config: %s", config)

	test_dataset = _dataset_from_config(config, args.test_split, args.limit_test)
	if getattr(test_dataset, "feature_dim", None) is None:
		raise RuntimeError("Dataset did not expose a feature dimension; check layer/token settings")

	input_dim = test_dataset.feature_dim  # type: ignore[assignment]
	model = OpeniaMLP(input_dim=input_dim, hidden_dim=config.hidden_dim) # type: ignore
	model.load_state_dict(state_dict)
	model.to(DEVICE)

	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.num_workers,
		pin_memory=(DEVICE.type == "cuda"),
	)

	criterion = torch.nn.BCEWithLogitsLoss()
	metrics = evaluate(model, test_loader, criterion)

	LOGGER.info("Test metrics: %s", metrics)
	for name, value in metrics.items():
		print(f"{name}: {value:.6f}")


if __name__ == "__main__":
	main()