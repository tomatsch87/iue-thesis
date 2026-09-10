from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from .utils import DEFAULT_DATA_ROOT, DEFAULT_PROJECT


@dataclass
class OpeniaConfig:
	data_root: Path = DEFAULT_DATA_ROOT
	train_split: str = "train"
	valid_split: str = "validation"
	layers: Sequence[int] = (48,)
	token_positions: Sequence[int] = (-2,)
	filter_empty: bool = False
	combiner: str = "concat"
	hidden_dim: Sequence[int] = (128, 64)
	batch_size: int = 32
	max_steps: int = 1000
	patience: Optional[int] = 0
	lr: float = 1e-3
	weight_decay: float = 0
	eval_every: int = 10
	log_every: int = 1
	num_workers: int = 0
	seed: int = 42
	output_dir: Path = Path("outputs/openia")
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
		base: "OpeniaConfig",
		override: Mapping[str, Any],
	) -> "OpeniaConfig":
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


class OpeniaMLP(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: Sequence[int]) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(input_dim, hidden_dim[0]),
			nn.ReLU(),
			nn.Linear(hidden_dim[0], hidden_dim[1]),
			nn.ReLU(),
			nn.Linear(hidden_dim[1], 1),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		logits = self.net(inputs)
		return logits.squeeze(-1)


OPENIA_SWEEP_CONFIG: Dict[str, Any] = {
	"method": "bayes",
	"metric": {"name": "val/roc_auc", "goal": "maximize"},
	"parameters": {
		"lr": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},
		"weight_decay": {"values": [0, 1e-1, 1e-2, 1e-3]},
		"hidden_dim": {"values": [[128, 64], [256, 128], [512, 256]]},
		"batch_size": {"values": [64, 128, 256]},
	},
}