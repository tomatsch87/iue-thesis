from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

try:
	from sklearn.metrics import f1_score, roc_auc_score
except ImportError:
	f1_score = None
	roc_auc_score = None

from .lstm import LineContextLSTM, LineLSTMConfig
from .sequence_dataset import LineSequenceHiddenStateDataset
from .utils import DEVICE, DEFAULT_DATA_ROOT, bss


LOGGER = logging.getLogger(__name__)


LineMeta = Tuple[str, str, int]


def _make_sequence_collate(
	batch: Sequence[Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	sequences, labels, _metadata = zip(*batch)
	lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
	padded_sequences = pad_sequence(list(sequences), batch_first=True, padding_value=0.0).to(torch.float32)
	stacked_labels = torch.stack(list(labels)).to(torch.float32)
	return padded_sequences, lengths, stacked_labels


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
	fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
	fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
	denom = 2 * tp + fp + fn
	if denom == 0:
		return 0.0
	return float((2 * tp) / denom)


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
	preds = (probs >= threshold).astype(np.float32)
	metrics: Dict[str, float] = {"accuracy": float((preds == y_true).mean())}

	if np.unique(y_true).size > 1:
		metrics["bss"] = float(bss(probs, y_true))
	else:
		metrics["bss"] = float("nan")

	if roc_auc_score is not None:
		try:
			metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
		except ValueError:
			metrics["roc_auc"] = float("nan")
	else:
		metrics["roc_auc"] = float("nan")

	if f1_score is not None:
		try:
			metrics["f1"] = float(f1_score(y_true, preds, pos_label=1))
		except ValueError:
			metrics["f1"] = float("nan")
	else:
		metrics["f1"] = _binary_f1(y_true.astype(np.int32), preds.astype(np.int32))

	return metrics


def _optimize_threshold_for_f1(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, Dict[str, float]]:
	thresholds = [i / 10.0 for i in range(1, 10)]
	best_threshold = thresholds[0]
	best_metrics = _compute_metrics(y_true, probs, best_threshold)
	best_f1 = best_metrics.get("f1", float("nan"))

	for threshold in thresholds[1:]:
		metrics = _compute_metrics(y_true, probs, threshold)
		f1 = metrics.get("f1", float("nan"))
		if np.isnan(best_f1) or (not np.isnan(f1) and f1 > best_f1):
			best_threshold = threshold
			best_metrics = metrics
			best_f1 = f1

	best_metrics = dict(best_metrics)
	best_metrics["best_threshold"] = float(best_threshold)
	return best_threshold, best_metrics


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate a line-sequence LSTM model on cached hidden-state features",
	)
	parser.add_argument("--model-path", type=Path, required=True, help="Path to trained LSTM checkpoint (.pt)")
	parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
	parser.add_argument("--test-jsonl", type=Path, required=True)
	parser.add_argument("--layers", type=int, nargs="*", default=None)
	parser.add_argument("--feature-splits", nargs="*", default=None)
	parser.add_argument("--include-ws-comment-tokens", action="store_true")
	parser.add_argument("--match-mode", choices=["auto", "index", "fingerprint"], default=None)
	parser.add_argument("--strict-mapping", action="store_true")
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--num-workers", type=int, default=0)
	return parser.parse_args()


def _dataset_from_config(args: argparse.Namespace, checkpoint_cfg: LineLSTMConfig) -> LineSequenceHiddenStateDataset:
	layers = tuple(sorted(set(args.layers))) if args.layers else checkpoint_cfg.layers
	feature_splits = tuple(args.feature_splits) if args.feature_splits else tuple(checkpoint_cfg.feature_splits)
	include_ws_comment_tokens = (
		True if args.include_ws_comment_tokens else bool(checkpoint_cfg.include_ws_comment_tokens)
	)
	match_mode = args.match_mode or checkpoint_cfg.match_mode
	strict_mapping = True if args.strict_mapping else bool(checkpoint_cfg.strict_mapping)

	return LineSequenceHiddenStateDataset(
		root=args.data_root,
		jsonl_path=args.test_jsonl,
		layers=layers,
		feature_splits=feature_splits,
		drop_ws_comment=True,
		include_ws_comment_tokens=include_ws_comment_tokens,
		match_mode=match_mode,
		strict_mapping=strict_mapping,
	)


def _build_model(config: LineLSTMConfig) -> LineContextLSTM:
	return LineContextLSTM(
		input_dim=config.input_dim,
		hidden_dim=config.hidden_dim,
		num_layers=config.lstm_layers,
		dropout=config.dropout,
		bidirectional=config.bidirectional,
	)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
	args = _parse_args()

	LOGGER.info("Loading checkpoint from %s", args.model_path)
	checkpoint = torch.load(args.model_path, map_location="cpu")
	checkpoint_config = checkpoint.get("config")
	if not isinstance(checkpoint_config, dict):
		raise RuntimeError("Checkpoint does not contain a valid 'config' dictionary")
	base_config = LineLSTMConfig.from_wandb_config(LineLSTMConfig(), checkpoint_config)

	test_dataset = _dataset_from_config(args, base_config)
	if test_dataset.feature_dim is None:
		raise RuntimeError("Unable to infer feature dimension from test dataset")

	model_config = LineLSTMConfig.from_wandb_config(base_config, {"input_dim": int(test_dataset.feature_dim)})
	model = _build_model(model_config).to(DEVICE)
	state_dict = checkpoint.get("state_dict")
	if not isinstance(state_dict, dict):
		raise RuntimeError("Checkpoint does not contain a valid 'state_dict'")
	model.load_state_dict(state_dict)
	model.eval()

	test_loader = DataLoader(
		test_dataset,
		batch_size=max(1, int(args.batch_size)),
		shuffle=False,
		num_workers=max(0, int(args.num_workers)),
		pin_memory=(DEVICE.type == "cuda"),
		collate_fn=_make_sequence_collate,
	)

	all_labels: List[float] = []
	all_probs: List[float] = []
	line_meta: List[LineMeta] = []

	dataset_offset = 0
	with torch.no_grad():
		for batch_idx, (inputs, lengths, labels) in enumerate(test_loader):
			_ = batch_idx
			inputs = inputs.to(DEVICE, non_blocking=True)
			lengths = lengths.to(DEVICE)
			labels = labels.to(DEVICE)

			logits = model(inputs, lengths)
			probs = torch.sigmoid(logits)
			all_labels.extend(labels.cpu().tolist())
			all_probs.extend(probs.cpu().tolist())

			for local_idx in range(inputs.size(0)):
				dataset_index = dataset_offset + local_idx
				if dataset_index >= len(test_dataset):
					continue
				item = test_dataset._items[dataset_index]  # pylint: disable=protected-access
				line_meta.append((str(item.path), str(item.candidate_key), int(item.line_number)))
			dataset_offset += inputs.size(0)

	y_true = np.asarray(all_labels, dtype=np.float32)
	probs = np.asarray(all_probs, dtype=np.float32)
	if y_true.size == 0:
		raise RuntimeError("No valid line labels found in evaluation set")

	best_threshold, metrics = _optimize_threshold_for_f1(y_true, probs)
	metrics["line_samples"] = float(len(line_meta))

	LOGGER.info("Best threshold by F1 from [0.1..0.9]: %.1f", best_threshold)
	LOGGER.info("Line-sequence test metrics at best threshold: %s", metrics)
	for name, value in metrics.items():
		print(f"{name}: {value:.6f}")


if __name__ == "__main__":
	main()