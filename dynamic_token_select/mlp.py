from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

from .utils import DEFAULT_DATA_ROOT, DEFAULT_PROJECT


@dataclass
class MLPConfig:
	data_root: Path = DEFAULT_DATA_ROOT
	train_split: str = "train"
	valid_split: str = "validation"
	layers: Optional[Sequence[int]] = (48,)
	filter_empty: bool = True
	token_filter: Optional[str] = None
	token_filter_config: Dict[str, Any] = field(default_factory=dict)
	tokenizer_name: Optional[str] = None
	input_dim: int = 2048
	k: int = 4
	mlp_layers: int = 2
	dropout: float = 0.2
	batch_size: int = 32
	max_steps: int = 1000
	patience: Optional[int] = 25
	lr: float = 1e-4
	weight_decay: float = 1e-4
	eval_every: int = 10
	log_every: int = 1
	num_workers: int = 0
	seed: int = 42
	output_dir: Path = Path("outputs/dynamic_token_mlp")
	wandb_project: str = DEFAULT_PROJECT
	run_name: Optional[str] = None
	sweep_name: Optional[str] = None
	gradient_clip_val: Optional[float] = None
	lr_scheduler: str = "cosine_warmup"
	warmup_steps: int = 50
	min_lr: float = 1e-6

	def to_logging_dict(self) -> Dict[str, Any]:
		payload = asdict(self)
		payload["data_root"] = str(self.data_root)
		payload["output_dir"] = str(self.output_dir)
		if self.layers is not None:
			payload["layers"] = list(self.layers)
		payload["token_filter_config"] = dict(self.token_filter_config)
		payload["lr_scheduler"] = str(self.lr_scheduler)
		return payload

	@classmethod
	def from_wandb_config(
		cls,
		base: "MLPConfig",
		override: Mapping[str, Any],
	) -> "MLPConfig":
		merged: Dict[str, Any] = dict(base.to_logging_dict())
		merged.update({k: v for k, v in override.items() if v is not None})

		merged["data_root"] = Path(str(merged["data_root"]))
		merged["output_dir"] = Path(str(merged["output_dir"]))
		if "layers" in merged and merged["layers"] is not None:
			merged["layers"] = tuple(int(x) for x in merged["layers"])
		else:
			merged["layers"] = None

		if merged.get("token_filter") is not None:
			merged["token_filter"] = str(merged["token_filter"]).lower()
		merged["token_filter_config"] = dict(merged.get("token_filter_config", {}))

		if merged.get("tokenizer_name") is not None:
			merged["tokenizer_name"] = str(merged["tokenizer_name"])

		merged["wandb_project"] = str(merged["wandb_project"])
		if merged.get("run_name") is not None:
			merged["run_name"] = str(merged["run_name"])
		if merged.get("sweep_name") is not None:
			merged["sweep_name"] = str(merged["sweep_name"])

		if merged.get("patience") is not None:
			merged["patience"] = int(merged["patience"])

		for field_name in (
			"input_dim",
			"k",
			"mlp_layers",
			"batch_size",
			"eval_every",
			"log_every",
			"num_workers",
			"warmup_steps",
		):
			if field_name in merged:
				merged[field_name] = int(merged[field_name])

		for float_field in (
			"lr",
			"weight_decay",
			"dropout",
			"gradient_clip_val",
			"min_lr",
		):
			if float_field in merged and merged[float_field] is not None:
				merged[float_field] = float(merged[float_field])

		if "lr_scheduler" in merged and merged["lr_scheduler"] is not None:
			merged["lr_scheduler"] = str(merged["lr_scheduler"]).lower()

		return cls(**merged)


class FixedTokenMLP(nn.Module):
	def __init__(
		self,
		*,
		per_token_dim: int,
		k: int,
		mlp_layers: int = 2,
		dropout: float = 0.1,
	) -> None:
		super().__init__()
		if k <= 0:
			raise ValueError("k must be positive for FixedTokenMLP")
		self.k = int(k)
		input_dim = per_token_dim * self.k
		if mlp_layers < 1:
			raise ValueError("mlp_layers must be at least 1")
		layers = []
		current_dim = input_dim
		for _ in range(mlp_layers):
			next_dim = max(1, current_dim // 2)
			layers.append(nn.Linear(current_dim, next_dim))
			layers.append(nn.ReLU())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			current_dim = next_dim
		layers.append(nn.Linear(current_dim, 1))
		self.net = nn.Sequential(*layers)

	def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
		if lengths.dim() != 1:
			lengths = lengths.view(-1)
		if torch.any(lengths != self.k):
			raise ValueError(
				f"FixedTokenMLP expects all sequences to have length {self.k}, received lengths={lengths.tolist()}"
			)
		if inputs.size(1) < self.k:
			raise ValueError(
				f"Input sequence shorter than expected: got {inputs.size(1)} < {self.k}"
			)
		trimmed = inputs[:, : self.k, :]
		flat = trimmed.reshape(trimmed.size(0), -1)
		logits = self.net(flat)
		return logits.squeeze(-1)


FIXED_TOKEN_MLP_SWEEP_CONFIG: Dict[str, Any] = {
	"method": "bayes",
	"metric": {"name": "val/roc_auc", "goal": "maximize"},
	"parameters": {
		"lr": {"min": 5e-5, "max": 5e-4, "distribution": "log_uniform_values"},
		"weight_decay": {"values": [1e-4, 5e-4]},
		"mlp_layers": {"values": [2, 3, 4]},
		"dropout": {"values": [0.1, 0.2, 0.3]},
		"k": {"values": [2, 4, 8]},
	},
}
