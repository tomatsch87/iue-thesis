from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Optional

import torch


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

DEFAULT_DATA_DIR_ENV_VAR: Final[str] = "IUE_DEFAULT_DATA_DIR"
DEFAULT_PROJECT_ENV_VAR: Final[str] = "IUE_PROJECT"
DEFAULT_DEVICE_ENV_VAR: Final[str] = "IUE_DEVICE"
CUDA_VISIBLE_DEVICES_ENV_VAR: Final[str] = "CUDA_VISIBLE_DEVICES"

DEFAULT_DATA_DIR: Final[Path] = Path(
    os.getenv(DEFAULT_DATA_DIR_ENV_VAR, str((PROJECT_ROOT.parent / "data").resolve()))
).expanduser()
DEFAULT_PROJECT: Final[str] = os.getenv(DEFAULT_PROJECT_ENV_VAR, "iue-thesis")
CUDA_VISIBLE_DEVICES: Final[Optional[str]] = os.getenv(CUDA_VISIBLE_DEVICES_ENV_VAR)


def _resolve_device() -> torch.device:
    requested_device = os.getenv(DEFAULT_DEVICE_ENV_VAR)
    if requested_device:
        try:
            candidate = torch.device(requested_device)
            if candidate.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return candidate
        except (TypeError, RuntimeError):
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE: Final[torch.device] = _resolve_device()
