from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .utils import DEFAULT_DATA_ROOT, DEFAULT_PROJECT


@dataclass
class LineLSTMConfig:
	data_root: Path = DEFAULT_DATA_ROOT
	train_jsonl: Optional[Path] = None
	valid_jsonl: Optional[Path] = None
	layers: Optional[Sequence[int]] = (48,)
	feature_splits: Sequence[str] = ("train", "validation", "test")
	drop_ws_comment: bool = True
	include_ws_comment_tokens: bool = False
	match_mode: str = "auto"
	strict_mapping: bool = False
	input_dim: int = 2048
	hidden_dim: int = 256
	lstm_layers: int = 1
	bidirectional: bool = False
	dropout: float = 0.1
	batch_size: int = 128
	max_steps: int = 1500
	patience: Optional[int] = 5
	lr: float = 2e-4
	pos_weight: Optional[float | str] = "auto"
	weight_decay: float = 0.0
	gradient_clip_val: Optional[float] = 1.0
	lr_scheduler: str = "none"
	warmup_steps: int = 0
	min_lr: float = 0.0
	eval_every: int = 50
	log_every: int = 20
	num_workers: int = 0
	seed: int = 42
	output_dir: Path = Path("outputs/token_level/line_lstm")
	save_best_model: bool = True
	save_best_per_run: bool = False
	wandb_project: str = DEFAULT_PROJECT
	run_name: Optional[str] = None
	sweep_name: Optional[str] = None

	def to_logging_dict(self) -> Dict[str, Any]:
		payload = asdict(self)
		payload["data_root"] = str(self.data_root)
		payload["output_dir"] = str(self.output_dir)
		payload["train_jsonl"] = str(self.train_jsonl) if self.train_jsonl is not None else None
		payload["valid_jsonl"] = str(self.valid_jsonl) if self.valid_jsonl is not None else None
		payload["feature_splits"] = list(self.feature_splits)
		if self.layers is not None:
			payload["layers"] = list(self.layers)
		return payload

	@classmethod
	def from_wandb_config(
		cls,
		base: "LineLSTMConfig",
		override: Mapping[str, Any],
	) -> "LineLSTMConfig":
		merged: Dict[str, Any] = dict(base.to_logging_dict())
		merged.update({k: v for k, v in override.items() if v is not None})

		merged["data_root"] = Path(str(merged["data_root"]))
		merged["output_dir"] = Path(str(merged["output_dir"]))
		if merged.get("train_jsonl") is not None:
			merged["train_jsonl"] = Path(str(merged["train_jsonl"]))
		if merged.get("valid_jsonl") is not None:
			merged["valid_jsonl"] = Path(str(merged["valid_jsonl"]))
		if "layers" in merged and merged["layers"] is not None:
			merged["layers"] = tuple(int(x) for x in merged["layers"])
		else:
			merged["layers"] = None
		if "feature_splits" in merged and merged["feature_splits"] is not None:
			merged["feature_splits"] = tuple(str(x) for x in merged["feature_splits"])

		for field_name in (
			"input_dim",
			"hidden_dim",
			"lstm_layers",
			"batch_size",
			"max_steps",
			"eval_every",
			"log_every",
			"num_workers",
			"seed",
			"warmup_steps",
		):
			if field_name in merged and merged[field_name] is not None:
				merged[field_name] = int(merged[field_name])

		if merged.get("patience") is not None:
			merged["patience"] = int(merged["patience"])

		for float_field in (
			"lr",
			"pos_weight",
			"weight_decay",
			"dropout",
			"gradient_clip_val",
			"min_lr",
		):
			if float_field in merged and merged[float_field] is not None and not (
				float_field == "pos_weight" and str(merged[float_field]).strip().lower() == "auto"
			):
				merged[float_field] = float(merged[float_field])

		for bool_field in (
			"drop_ws_comment",
			"include_ws_comment_tokens",
			"strict_mapping",
			"bidirectional",
			"save_best_model",
			"save_best_per_run",
		):
			if bool_field in merged:
				value = merged[bool_field]
				if isinstance(value, str):
					value = value.strip().lower() in {"1", "true", "yes", "y"}
				merged[bool_field] = bool(value)

		for str_field in (
			"match_mode",
			"lr_scheduler",
			"wandb_project",
			"run_name",
			"sweep_name",
		):
			if str_field in merged and merged[str_field] is not None:
				merged[str_field] = str(merged[str_field])
		if merged.get("lr_scheduler") is not None:
			merged["lr_scheduler"] = str(merged["lr_scheduler"]).lower()

		return cls(**merged)


class LineContextLSTM(nn.Module):
	def __init__(
		self,
		input_dim: int,
		hidden_dim: int,
		*,
		num_layers: int = 2,
		dropout: float = 0.1,
		bidirectional: bool = False,
	) -> None:
		super().__init__()
		lstm_dropout = dropout if num_layers > 1 else 0.0
		self.lstm = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=lstm_dropout,
			bidirectional=bidirectional,
		)
		self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
		self.output_dim = hidden_dim * (2 if bidirectional else 1)
		self.classifier = nn.Linear(self.output_dim, 1)

	def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
		if lengths.dim() != 1:
			lengths = lengths.view(-1)
		packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
		_, (hidden, _) = self.lstm(packed)
		if self.lstm.bidirectional:
			last_forward = hidden[-2]
			last_backward = hidden[-1]
			features = torch.cat([last_forward, last_backward], dim=-1)
		else:
			features = hidden[-1]
		logits = self.classifier(self.dropout(features)).squeeze(-1)
		return logits


LINE_LSTM_SWEEP_CONFIG: Dict[str, Any] = {
	"method": "bayes",
	"metric": {"name": "val/f1", "goal": "maximize"},
	"parameters": {
		"lr": {"min": 1e-5, "max": 5e-4, "distribution": "log_uniform_values"},
		"lstm_layers": {"values": [1, 2]},
		"hidden_dim": {"values": [128, 256, 512]},
		"include_ws_comment_tokens": {"values": [False, True]},
	},
}
