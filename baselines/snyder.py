from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from .utils import DEFAULT_DATA_ROOT, DEFAULT_PROJECT


@dataclass
class SnyderConfig:
	data_root: Path = DEFAULT_DATA_ROOT
	train_split: str = "train"
	valid_split: str = "validation"
	layers: Sequence[int] = (48,)
	token_positions: Sequence[int] = (0,)
	filter_empty: bool = False
	combiner: str = "concat"
	hidden_dim: int = 256
	batch_size: int = 128
	max_steps: int = 1000
	patience: Optional[int] = 0
	lr: float = 1e-4
	weight_decay: float = 1e-2
	eval_every: int = 10
	log_every: int = 1
	num_workers: int = 0
	seed: int = 42
	output_dir: Path = Path("outputs/snyder")
	wandb_project: str = DEFAULT_PROJECT
	run_name: Optional[str] = None
	sweep_name: Optional[str] = None

	def to_logging_dict(self) -> Dict[str, object]:
		payload = asdict(self)
		payload["data_root"] = str(self.data_root)
		payload["output_dir"] = str(self.output_dir)
		payload["layers"] = list(self.layers)
		payload["token_positions"] = list(self.token_positions)
		return payload

	@classmethod
	def from_wandb_config(
		cls,
		base: "SnyderConfig",
		override: Mapping[str, Any],
	) -> "SnyderConfig":
		merged: Dict[str, Any] = dict(base.to_logging_dict())
		merged.update({k: v for k, v in override.items() if v is not None})

		merged["data_root"] = Path(str(merged["data_root"]))
		merged["output_dir"] = Path(str(merged["output_dir"]))

		layers_raw = merged.get("layers", base.layers)
		token_positions_raw = merged.get("token_positions", base.token_positions)
		merged["layers"] = tuple(int(x) for x in layers_raw)  # type: ignore[arg-type]
		merged["token_positions"] = tuple(int(x) for x in token_positions_raw)  # type: ignore[arg-type]

		if "filter_empty" in merged:
			merged["filter_empty"] = bool(merged["filter_empty"])

		merged["wandb_project"] = str(merged["wandb_project"])
		if merged.get("run_name") is not None:
			merged["run_name"] = str(merged["run_name"])
		if merged.get("sweep_name") is not None:
			merged["sweep_name"] = str(merged["sweep_name"])

		if merged.get("patience") is not None:
			merged["patience"] = int(merged["patience"])

		if merged.get("combiner") is not None:
			merged["combiner"] = str(merged["combiner"]).lower()

		return cls(**merged)


class SnyderMLP(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		logits = self.net(inputs)
		return logits.squeeze(-1)


SNYDER_SWEEP_CONFIG: Dict[str, Any] = {
	"method": "bayes",
	"metric": {"name": "val/roc_auc", "goal": "maximize"},
	"parameters": {
		"lr": {"min": 1e-6, "max": 1e-3, "distribution": "log_uniform_values"},
		"weight_decay": {"values": [1e-1, 1e-2, 1e-3]},
		"hidden_dim": {"values": [128, 256, 512]},
		"batch_size": {"values": [128, 256, 512]},
	},
}