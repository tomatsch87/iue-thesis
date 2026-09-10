from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .utils import DEFAULT_DATA_ROOT, DEFAULT_PROJECT


@dataclass
class LSTMConfig:
	data_root: Path = DEFAULT_DATA_ROOT
	train_split: str = "train"
	valid_split: str = "validation"
	layers: Optional[Sequence[int]] = (48,)
	filter_empty: bool = True
	token_filter: Optional[str] = None
	token_filter_config: Dict[str, Any] = field(default_factory=dict)
	tokenizer_name: Optional[str] = None
	input_dim: int = 2048
	hidden_dim: int = 256
	lstm_layers: int = 1
	dropout: float = 0
	batch_size: int = 128
	max_steps: int = 1000
	patience: Optional[int] = 15
	lr: float = 1e-4
	weight_decay: float = 0
	lr_scheduler: str = "none"
	warmup_steps: int = 0
	min_lr: float = 0.0
	eval_every: int = 10
	log_every: int = 1
	num_workers: int = 0
	seed: int = 42
	output_dir: Path = Path("outputs/dynamic_token_lstm")
	wandb_project: str = DEFAULT_PROJECT
	run_name: Optional[str] = None
	sweep_name: Optional[str] = None

	def to_logging_dict(self) -> Dict[str, Any]:
		payload = asdict(self)
		payload["data_root"] = str(self.data_root)
		payload["output_dir"] = str(self.output_dir)
		if self.layers is not None:
			payload["layers"] = list(self.layers)
		payload["token_filter_config"] = dict(self.token_filter_config)
		return payload

	@classmethod
	def from_wandb_config(
		cls,
		base: "LSTMConfig",
		override: Mapping[str, Any],
	) -> "LSTMConfig":
		merged: Dict[str, Any] = dict(base.to_logging_dict())
		merged.update({k: v for k, v in override.items() if v is not None})

		merged["data_root"] = Path(str(merged["data_root"]))
		merged["output_dir"] = Path(str(merged["output_dir"]))
		if "layers" in merged and merged["layers"] is not None:
			merged["layers"] = tuple(int(x) for x in merged["layers"])  # type: ignore[arg-type]
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
			"lstm_layers",
			"hidden_dim",
			"batch_size",
			"eval_every",
			"log_every",
			"num_workers",
			"encoder_hidden_dim",
			"encoder_output_dim",
			"encoder_layers",
			"att_dim",
			"warmup_steps",
		):
			if field_name in merged:
				merged[field_name] = int(merged[field_name])

		for float_field in (
			"lr",
			"weight_decay",
			"dropout",
			"encoder_dropout",
			"min_lr",
		):
			if float_field in merged and merged[float_field] is not None:
				merged[float_field] = float(merged[float_field])

		if "gated_attention" in merged:
			value = merged["gated_attention"]
			if isinstance(value, str):
				value = value.strip().lower() in {"1", "true", "yes", "y"}
			merged["gated_attention"] = bool(value)

		if "lr_scheduler" in merged and merged["lr_scheduler"] is not None:
			merged["lr_scheduler"] = str(merged["lr_scheduler"]).lower()

		return cls(**merged)


class DynamicTokenLSTM(nn.Module):
	def __init__(
		self,
		input_dim: int,
		hidden_dim: int,
		*,
		num_layers: int = 1,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		lstm_dropout = dropout if num_layers > 1 else 0.0
		self.lstm = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=lstm_dropout,
		)
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
		self.classifier = nn.Linear(hidden_dim, 1)

	def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
		if lengths.dim() != 1:
			lengths = lengths.view(-1)
		packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
		_, (hidden, _) = self.lstm(packed)
		features = self._final_hidden(hidden)
		logit = self.classifier(self.dropout(features))
		return logit.squeeze(-1)

	def _final_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
		return hidden[-1]


DYNAMIC_TOKEN_LSTM_SWEEP_CONFIG: Dict[str, Any] = {
	"method": "bayes",
	"metric": {"name": "val/roc_auc", "goal": "maximize"},
	"parameters": {
		"lr": {"min": 1e-5, "max": 5e-4, "distribution": "log_uniform_values"},
		"weight_decay": {"values": [0, 1e-2, 1e-4]},
		"hidden_dim": {"values": [128, 256, 512]},
		"dropout": {"values": [0.0, 0.1, 0.2]},
		"batch_size": {"values": [32, 64, 128]},
	},
}
