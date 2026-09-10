from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from global_utils import bss, seed_everything
from global_variables import DEFAULT_DATA_DIR, DEFAULT_PROJECT, DEVICE


LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_ROOT: Final[Path] = DEFAULT_DATA_DIR / "feats_lcb_qwen3"

__all__ = ["DEFAULT_DATA_ROOT", "DEFAULT_PROJECT", "DEVICE", "bss", "seed_everything"]
