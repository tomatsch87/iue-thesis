"""
Dataset utilities for baseline training/evaluation over extracted `.pt` features.

Default dataset behavior:
- `HiddenStateDataset` reads `<root>/<split>/*.pt`, finds candidates with `is_correct`,
	and returns `(features, label)` per candidate.
- Feature vectors are built from selected `layers` x `token_positions` using combiner
	`concat` by default (or `average`).
- `filter_empty=False` by default for `HiddenStateDataset`.

Additional dataset options:
- `token_positions` supports positive indices and limited negative indexing near the tail.
- `combiner`: `concat` (flatten all vectors) or `average` (same-shape vectors only).
- `HiddenStateRandomTailDataset` samples one token from the tail of the code segment:
	defaults are `filter_empty=True`, `layer=<required>`, and `tail_fraction=0.5`.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SampleIndex:
	path: Path
	candidate_idx: int
	label: float


@dataclass(frozen=True)
class TailSampleIndex:
	path: Path
	candidate_idx: int
	label: float
	start_pos: int
	end_pos: int


class HiddenStateDataset(Dataset):
	def __init__(
		self,
		root: Path,
		split: str,
		layers: Sequence[int],
		token_positions: Sequence[int],
		filter_empty: bool = False,
		combiner: Optional[str] = None,
	) -> None:
		self.root = root
		self.split = split
		self.layers = tuple(layers)
		self.token_positions = tuple(token_positions)
		self.filter_empty = filter_empty
		self._vector_target_count = len(self.layers) * len(self.token_positions)
		self.combiner = (combiner or "concat").lower()
		allowed_combiners = {"concat", "average"}
		if self.combiner not in allowed_combiners:
			raise ValueError(f"Unsupported combiner '{combiner}'. Choose from {sorted(allowed_combiners)}")
		self._items: List[SampleIndex] = []
		self.feature_dim: Optional[int] = None
		self._cache_path: Optional[Path] = None
		self._cache_payload: Optional[Dict[str, Any]] = None

		split_dir = self.root / self.split
		if not split_dir.exists():
			raise FileNotFoundError(f"Split directory not found: {split_dir}")

		files = sorted(split_dir.glob("*.pt"))
		if not files:
			raise FileNotFoundError(f"No .pt files found in {split_dir}")

		skipped_no_label = 0
		skipped_missing_feature = 0

		for file_path in files:
			payload: Dict[str, Any] = torch.load(file_path, map_location="cpu")
			feature_dict = payload.get("features", {})
			if not isinstance(feature_dict, dict):
				continue
			for cand_idx, candidate in feature_dict.items():
				label = candidate.get("is_correct")
				if label is None:
					skipped_no_label += 1
					continue
				if not self._candidate_has_required_features(candidate):
					skipped_missing_feature += 1
					continue
				if self.feature_dim is None:
					self.feature_dim = self._infer_feature_dim(candidate)
				self._items.append(
					SampleIndex(path=file_path, candidate_idx=int(cand_idx), label=float(label))
				)

		if self.feature_dim is None or not self._items:
			raise RuntimeError(
				"No usable samples found; verify layer/token parameters against the stored features."
			)

		if skipped_no_label or skipped_missing_feature:
			LOGGER.info(
				"Loaded split %s with %s samples (skipped %s without labels, %s without features)",
				split,
				len(self._items),
				skipped_no_label,
				skipped_missing_feature,
			)
		else:
			LOGGER.info("Loaded split %s with %s samples", split, len(self._items))

	def _candidate_has_required_features(self, candidate: Mapping[str, Any]) -> bool:
		if self.filter_empty:
			code_token_idx = candidate.get("code_token_idx")
			if (
				isinstance(code_token_idx, (tuple, list))
				and len(code_token_idx) == 2
				and all(isinstance(idx, int) for idx in code_token_idx)
			):
				start_idx, end_idx = int(code_token_idx[0]), int(code_token_idx[1])
				if start_idx == 0 and end_idx == 0:
					return False
		hidden_states = candidate.get("hidden_states")
		if not isinstance(hidden_states, dict):
			return False
		for layer in self.layers:
			layer_dict = hidden_states.get(layer)
			if not isinstance(layer_dict, dict):
				return False
			sorted_keys = self._sorted_token_keys(layer_dict)
			for token_pos in self.token_positions:
				resolved_key = self._resolve_token_key(layer_dict, sorted_keys, token_pos)
				if resolved_key is None:
					return False
		return True

	def _infer_feature_dim(self, candidate: Mapping[str, Any]) -> int:
		hidden_states = candidate.get("hidden_states")
		assert isinstance(hidden_states, dict)
		vectors = self._collect_vectors(hidden_states)
		features = self._combine_vectors(vectors)
		return int(features.numel())

	def __len__(self) -> int:
		return len(self._items)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		item = self._items[index]
		payload = self._load_payload(item.path)
		features_by_candidate = payload["features"]
		candidate = features_by_candidate[item.candidate_idx]
		hidden_states = candidate["hidden_states"]

		vectors = self._collect_vectors(hidden_states, context=item.path.name)
		features = self._combine_vectors(vectors)
		label = torch.tensor(item.label, dtype=torch.float32)
		return features, label
	def _collect_vectors(
		self,
		hidden_states: Mapping[Any, Any],
		*,
		context: Optional[str] = None,
	) -> List[torch.Tensor]:
		vectors: List[torch.Tensor] = []
		for layer in self.layers:
			layer_dict = hidden_states[layer]
			sorted_keys = self._sorted_token_keys(layer_dict)
			for token_pos in self.token_positions:
				resolved_key = self._resolve_token_key(layer_dict, sorted_keys, token_pos)
				if resolved_key is None:
					location = f" in {context}" if context else ""
					raise KeyError(
						f"Token position {token_pos} not available for layer {layer}{location}"
					)
				vector = layer_dict[resolved_key].to(torch.float32)
				vectors.append(vector)
		return vectors

	def _combine_vectors(self, vectors: Sequence[torch.Tensor]) -> torch.Tensor:
		vec_list = list(vectors)
		if not vec_list:
			raise RuntimeError("No vectors collected to combine.")
		if self.combiner == "concat":
			return vec_list[0] if len(vec_list) == 1 else torch.cat(vec_list, dim=0)
		if self.combiner == "average":
			if len(vec_list) == 1:
				return vec_list[0]
			ref_shape = vec_list[0].shape
			for idx, vector in enumerate(vec_list[1:], start=1):
				if vector.shape != ref_shape:
					raise ValueError(
						f"All vectors must share the same shape for average combiner;"
						f" got {ref_shape} and {vector.shape} at index {idx}."
				)
			stacked = torch.stack(vec_list, dim=0)
			return stacked.mean(dim=0)
		raise RuntimeError(f"Unhandled combiner '{self.combiner}'.")

	def _load_payload(self, path: Path) -> Dict[str, Any]:
		if self._cache_path == path and self._cache_payload is not None:
			return self._cache_payload
		payload: Dict[str, Any] = torch.load(path, map_location="cpu")
		self._cache_path = path
		self._cache_payload = payload
		return payload

	def _sorted_token_keys(self, layer_dict: Mapping[Any, Any]) -> List[Any]:
		try:
			return sorted(layer_dict.keys(), key=int)
		except (TypeError, ValueError):
			return sorted(layer_dict.keys())

	def _resolve_token_key(
		self,
		layer_dict: Mapping[Any, Any],
		sorted_keys: Sequence[Any],
		token_pos: int,
	) -> Optional[Any]:
		if token_pos >= 0:
			return token_pos if token_pos in layer_dict else None
		key_count = len(sorted_keys)
		if key_count <= 1 or key_count > 4:
			return None
		# handle edge case for key_count = 3 and token_pos = -3
		# in the case where there are 3 tokens, -3 should map to the first token
		if key_count == 3 and token_pos == -3:
			return sorted_keys[0]
		max_supported_neg = min(key_count - 1, 3)
		if -token_pos > max_supported_neg:
			return None
		return sorted_keys[token_pos]


class HiddenStateRandomTailDataset(Dataset):
	def __init__(
		self,
		root: Path,
		split: str,
		layer: int,
		filter_empty: bool = True,
		tail_fraction: float = 0.5,
	) -> None:
		if not (0.0 < tail_fraction <= 1.0):
			raise ValueError("tail_fraction must be in (0.0, 1.0].")
		self.root = root
		self.split = split
		self.filter_empty = filter_empty
		self._layer = int(layer)
		self._tail_fraction = float(tail_fraction)
		self._items: List[TailSampleIndex] = []
		self.hidden_size: Optional[int] = None
		self._cache_path: Optional[Path] = None
		self._cache_payload: Optional[Dict[str, Any]] = None

		split_dir = self.root / self.split
		if not split_dir.exists():
			raise FileNotFoundError(f"Split directory not found: {split_dir}")

		files = sorted(split_dir.glob("*.pt"))
		if not files:
			raise FileNotFoundError(f"No .pt files found in {split_dir}")

		skipped_no_label = 0
		skipped_missing_sequence = 0
		skipped_missing_layer = 0

		for file_path in files:
			payload: Dict[str, Any] = torch.load(file_path, map_location="cpu")
			feature_dict = payload.get("features", {})
			if not isinstance(feature_dict, dict):
				continue
			for cand_idx, candidate in feature_dict.items():
				label = candidate.get("is_correct")
				if label is None:
					skipped_no_label += 1
					continue
				bounds = self._determine_sequence_bounds(candidate)
				if bounds is None:
					skipped_missing_sequence += 1
					continue
				start_pos, end_pos = bounds
				hidden_states = candidate.get("hidden_states")
				if not isinstance(hidden_states, dict) or not hidden_states:
					skipped_missing_layer += 1
					continue
				if not self._candidate_has_required_tokens(
					hidden_states,
					start_pos,
					end_pos,
				):
					skipped_missing_sequence += 1
					continue
				if self.hidden_size is None:
					self.hidden_size = self._infer_hidden_size(hidden_states)
				self._items.append(
					TailSampleIndex(
						path=file_path,
						candidate_idx=int(cand_idx),
						label=float(label),
						start_pos=start_pos,
						end_pos=end_pos,
					)
				)

		if not self._items:
			raise RuntimeError("No usable samples found; verify that hidden states include the requested sequence.")
		if self.hidden_size is None:
			raise RuntimeError("Unable to infer hidden size from the provided features.")

		if skipped_no_label or skipped_missing_sequence or skipped_missing_layer:
			LOGGER.info(
				"Loaded split %s with %s samples (skipped %s without labels, %s without sequences, %s without layer)",
				split,
				len(self._items),
				skipped_no_label,
				skipped_missing_sequence,
				skipped_missing_layer,
			)
		else:
			LOGGER.info("Loaded split %s with %s samples", split, len(self._items))

	def __len__(self) -> int:
		return len(self._items)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		item = self._items[index]
		payload = self._load_payload(item.path)
		candidate = payload["features"][item.candidate_idx]
		hidden_states = candidate["hidden_states"]
		sequence = self._build_sequence(hidden_states, item.start_pos, item.end_pos)
		if sequence.size(0) == 0:
			raise RuntimeError("Empty hidden-state sequence encountered.")
		seq_len = sequence.size(0)
		count = max(1, int(math.ceil(seq_len * self._tail_fraction)))
		start_idx = max(0, seq_len - count)
		chosen_idx = random.randint(start_idx, seq_len - 1)
		selected = sequence[chosen_idx]
		if selected.size(0) == 1:
			selected = selected[0]
		label = torch.tensor(item.label, dtype=torch.float32)
		return selected, label

	def _build_sequence(
		self,
		hidden_states: Mapping[Any, Any],
		start_pos: int,
		end_pos: int,
	) -> torch.Tensor:
		positions = range(start_pos, end_pos + 1)
		per_token: List[torch.Tensor] = []
		layer_dict = hidden_states.get(self._layer)
		if not isinstance(layer_dict, Mapping):
			raise KeyError(f"Layer {self._layer} missing for sequence construction")
		for pos in positions:
			vector = self._get_vector(layer_dict, pos)
			if vector is None:
				raise KeyError(f"Token position {pos} missing for layer {self._layer}")
			per_token.append(vector.to(torch.float32))
		return torch.stack(per_token, dim=0)

	def _candidate_has_required_tokens(
		self,
		hidden_states: Mapping[Any, Any],
		start_pos: int,
		end_pos: int,
	) -> bool:
		layer_dict = hidden_states.get(self._layer)
		if not isinstance(layer_dict, Mapping):
			return False
		for idx in range(start_pos, end_pos + 1):
			if self._get_vector(layer_dict, idx) is None:
				return False
		return True

	def _infer_hidden_size(self, hidden_states: Mapping[Any, Any]) -> int:
		layer_dict = hidden_states.get(self._layer)
		if not isinstance(layer_dict, Mapping):
			raise RuntimeError("Unable to infer hidden size from hidden state tensors")
		first_vector = next(iter(layer_dict.values()), None)
		if isinstance(first_vector, torch.Tensor):
			return int(first_vector.numel())
		raise RuntimeError("Unable to infer hidden size from hidden state tensors")

	def _determine_sequence_bounds(self, candidate: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
		code_range = candidate.get("code_token_idx")
		assistant_token_count = candidate.get("assistant_token_count")
		if (
			not isinstance(code_range, (list, tuple))
			or len(code_range) < 2
			or assistant_token_count is None
		):
			return None
		try:
			code_start = int(code_range[0])
			code_end = int(code_range[1])
			assistant_tokens = int(assistant_token_count)
		except (TypeError, ValueError):
			return None
		if self.filter_empty and code_start == 0 and code_end == 0:
			return None
		penultimate = assistant_tokens - 2
		if penultimate < code_start:
			return None
		return code_start, penultimate


	@staticmethod
	def _get_vector(layer_dict: Mapping[Any, Any], token_idx: int) -> Optional[torch.Tensor]:
		if token_idx in layer_dict:
			vector = layer_dict[token_idx]
			return vector if isinstance(vector, torch.Tensor) else None
		str_key = str(token_idx)
		vector = layer_dict.get(str_key)
		return vector if isinstance(vector, torch.Tensor) else None

	def _load_payload(self, path: Path) -> Dict[str, Any]:
		if self._cache_path == path and self._cache_payload is not None:
			return self._cache_payload
		payload: Dict[str, Any] = torch.load(path, map_location="cpu")
		self._cache_path = path
		self._cache_payload = payload
		return payload

	@staticmethod
	def _safe_int(value: Any) -> Optional[int]:
		if isinstance(value, torch.Tensor):
			if value.numel() != 1:
				return None
			value = value.item()
		try:
			return int(value)
		except (TypeError, ValueError):
			return None
