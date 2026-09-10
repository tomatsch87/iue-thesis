from __future__ import annotations

import os
import random

import numpy as np
import torch

from global_variables import DEVICE


def bss(confidences:np.ndarray, labels:np.ndarray) -> float:
	"""
	Calculate the Brier Skill Score (BSS).

	Baseline score is 0.0. Negative scores show deterioration,
	positive scores show improvement over the baseline score,
	BSS=1.0 is a perfect prediction.

	Args:
		confidences: List of prediction probabilities
		labels: List of labels indicating correctness

	Returns:
		BSS score
	"""
	brier_score = np.mean((labels - confidences) ** 2)
	base_rate = np.mean(labels)
	brier_ref = base_rate * (1 - base_rate)
	return float((brier_ref - brier_score) / brier_ref)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return DEVICE
