from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

try:
    from torchmil.models import ABMIL as TorchMILABMIL
except ImportError as exc:
    raise ImportError("torchmil must be installed to use the ABMIL model") from exc

from dynamic_token_select.utils import DEFAULT_DATA_ROOT, DEFAULT_PROJECT

LOGGER = logging.getLogger(__name__)


@dataclass
class ABMILConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    train_split: str = "train"
    valid_split: str = "validation"
    layers: Optional[Sequence[int]] = (48,)
    filter_empty: bool = True
    token_filter: Optional[str] = None
    token_filter_config: Dict[str, Any] = field(default_factory=dict)
    tokenizer_name: Optional[str] = None
    input_dim: int = 2048
    batch_size: int = 32
    max_steps: int = 1000
    patience: Optional[int] = 25
    lr: float = 1e-4
    weight_decay: float = 1e-4
    eval_every: int = 10
    log_every: int = 1
    num_workers: int = 0
    seed: int = 42
    output_dir: Path = Path("outputs/abmil")
    wandb_project: str = DEFAULT_PROJECT
    run_name: Optional[str] = None
    sweep_name: Optional[str] = None
    gradient_clip_val: Optional[float] = 1.0
    lr_scheduler: str = "cosine_warmup"
    warmup_steps: int = 50
    min_lr: float = 1e-6
    # ABMIL parameters
    encoder_hidden_dim: int = 256
    encoder_output_dim: int = 128
    encoder_layers: int = 2
    encoder_dropout: float = 0.1
    att_dim: int = 64
    att_act: str = "tanh"
    gated_attention: bool = True

    def to_logging_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["output_dir"] = str(self.output_dir)
        if self.layers is not None:
            payload["layers"] = list(self.layers)
        if self.gradient_clip_val is not None:
            payload["gradient_clip_val"] = self.gradient_clip_val
        payload["token_filter_config"] = dict(self.token_filter_config)
        payload["lr_scheduler"] = str(self.lr_scheduler)
        return payload

    @classmethod
    def from_wandb_config(
        cls,
        base: "ABMILConfig",
        override: Mapping[str, Any],
    ) -> "ABMILConfig":
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

        int_fields = (
            "input_dim",
            "batch_size",
            "eval_every",
            "log_every",
            "num_workers",
            "encoder_hidden_dim",
            "encoder_output_dim",
            "encoder_layers",
            "att_dim",
            "warmup_steps",
        )
        for field_name in int_fields:
            if field_name in merged:
                merged[field_name] = int(merged[field_name])

        float_fields = (
            "lr",
            "weight_decay",
            "encoder_dropout",
            "gradient_clip_val",
            "min_lr",
        )
        for float_field in float_fields:
            if float_field in merged and merged[float_field] is not None:
                merged[float_field] = float(merged[float_field])

        if "gated_attention" in merged:
            value = merged["gated_attention"]
            if isinstance(value, str):
                value = value.strip().lower() in {"1", "true", "yes", "y"}
            merged["gated_attention"] = bool(value)

        if "att_act" in merged and merged["att_act"] is not None:
            merged["att_act"] = str(merged["att_act"]).lower()

        if "lr_scheduler" in merged and merged["lr_scheduler"] is not None:
            merged["lr_scheduler"] = str(merged["lr_scheduler"]).lower()

        return cls(**merged)


class InstanceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(max(num_hidden_layers - 1, 0)):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ABMILBagModel(nn.Module):
    def __init__(self, config: ABMILConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = InstanceEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.encoder_hidden_dim,
            output_dim=config.encoder_output_dim,
            num_hidden_layers=config.encoder_layers,
            dropout=config.encoder_dropout,
        )
        self.mil = TorchMILABMIL(
            in_shape=(config.encoder_output_dim,),
            att_dim=config.att_dim,
            att_act=config.att_act,
            gated=config.gated_attention,
            feat_ext=nn.Identity(),
            criterion=nn.BCEWithLogitsLoss(),
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        mask = self._lengths_to_mask(lengths, inputs.size(1), inputs.device)
        logits = self.mil(encoded, mask=mask)
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        return logits

    @staticmethod
    def _lengths_to_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        if lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)
        arange = torch.arange(max_len, device=device).unsqueeze(0)
        mask = (arange < lengths.unsqueeze(1)).int()
        return mask


ABMIL_SWEEP_CONFIG: Dict[str, Any] = {
    "method": "bayes",
    "metric": {"name": "val/roc_auc", "goal": "maximize"},
    "parameters": {
        "lr": {"min": 5e-5, "max": 5e-4, "distribution": "log_uniform_values"},
        "weight_decay": {"values": [1e-4, 5e-4]},
        "encoder_hidden_dim": {"values": [64, 128, 256]},
        "encoder_layers": {"values": [2, 3, 4]},
        "encoder_dropout": {"values": [0.1, 0.2, 0.3]},
        "att_dim": {"values": [64, 128]},
        "gated_attention": {"values": [False, True]},
    },
}