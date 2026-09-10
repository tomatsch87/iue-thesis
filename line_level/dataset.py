"""
Line-level dataset that aggregates token labels/features into one sample per line.

Default dataset behavior:
- Scans feature files from `feature_splits=('train', 'validation', 'test')`.
- Aligns JSONL candidates to feature candidates (`match_mode='auto'` by default).
- Drops whitespace/comment tokens by default (`drop_ws_comment=True`).
- Produces one sample per line with `(features, label, metadata)`.
- Label convention is hallucination-positive: clean/correct line -> `0.0`,
  line with at least one error -> `1.0`.

Additional options:
- `line_representation='last_token'` (default) or `aggregate` over line tokens.
- `aggregation='mean'` (default) or `max` when using aggregate representation.
- `layers` can be fixed or inferred; `combiner` is `concat` by default (`stack` optional).
- Mapping controls: `strict_mapping`, `warn_on_collision`, `expected_jsonl_split`,
  and `split_priority`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureEntry:
	path: Path
	split: str
	sample_id: str


@dataclass(frozen=True)
class LineSampleIndex:
	path: Path
	candidate_key: Hashable
	line_number: int
	label: float
	token_positions: Tuple[int, ...]


class LineLevelHiddenStateDataset(Dataset):
	def __init__(
		self,
		root: Path,
		jsonl_path: Path,
		layers: Optional[Sequence[int]] = None,
		feature_splits: Sequence[str] = ("train", "validation", "test"),
		drop_ws_comment: bool = True,
		combiner: str = "concat",
		match_mode: str = "auto",
		expected_jsonl_split: Optional[str] = None,
		strict_mapping: bool = False,
		warn_on_collision: bool = True,
		split_priority: Optional[Sequence[str]] = None,
		line_representation: str = "last_token",
		aggregation: str = "mean",
	) -> None:
		self.root = Path(root)
		self.jsonl_path = Path(jsonl_path)
		self.drop_ws_comment = bool(drop_ws_comment)
		self.strict_mapping = bool(strict_mapping)
		self.warn_on_collision = bool(warn_on_collision)
		self.feature_splits = tuple(self._canonical_split_name(name) for name in feature_splits)
		self._split_priority = tuple(
			self._canonical_split_name(name) for name in (split_priority or ("train", "validation", "test"))
		)
		self.expected_jsonl_split = (
			self._canonical_split_name(expected_jsonl_split) if expected_jsonl_split else None
		)
		self.line_representation = (line_representation or "last_token").lower()
		if self.line_representation not in {"last_token", "aggregate"}:
			raise ValueError("Unsupported line_representation. Expected one of {'last_token', 'aggregate'}.")
		self.aggregation = (aggregation or "mean").lower()
		if self.aggregation not in {"mean", "max"}:
			raise ValueError("Unsupported aggregation. Expected one of {'mean', 'max'}.")

		self.layers: Optional[Tuple[int, ...]] = tuple(int(layer) for layer in layers) if layers is not None else None
		self.layer_count: Optional[int] = None if self.layers is None else len(self.layers)
		self.hidden_size: Optional[int] = None
		self.feature_dim: Optional[int] = None
		self.combiner = (combiner or "concat").lower()
		if self.combiner not in {"concat", "stack"}:
			raise ValueError("Unsupported combiner. Expected one of {'concat', 'stack'}.")
		self.match_mode = (match_mode or "auto").lower()
		if self.match_mode not in {"auto", "index", "fingerprint"}:
			raise ValueError("Unsupported match_mode. Expected one of {'auto', 'index', 'fingerprint'}.")

		if not self.root.exists():
			raise FileNotFoundError(f"Feature root not found: {self.root}")
		if not self.jsonl_path.exists():
			raise FileNotFoundError(f"JSONL path not found: {self.jsonl_path}")

		self._items: List[LineSampleIndex] = []
		self._feature_index: Dict[str, List[FeatureEntry]] = {}
		self._payload_cache_path: Optional[Path] = None
		self._payload_cache: Optional[Dict[str, Any]] = None
		self._token_hash_cache: Dict[Path, Dict[str, List[Hashable]]] = {}
		self._collision_warned: set[str] = set()

		self.stats: Dict[str, int] = {
			"feature_files_scanned": 0,
			"feature_files_loaded": 0,
			"feature_files_skipped": 0,
			"samples_missing_features": 0,
			"feature_collisions": 0,
			"jsonl_rows_total": 0,
			"jsonl_rows_usable": 0,
			"jsonl_candidates_total": 0,
			"jsonl_candidates_matched": 0,
			"jsonl_candidates_unmatched": 0,
			"jsonl_candidates_invalid": 0,
			"line_labels_total": 0,
			"line_labels_kept": 0,
			"lines_dropped_missing_label": 0,
			"tokens_dropped_ws_comment": 0,
			"tokens_dropped_no_line": 0,
			"tokens_dropped_out_of_code_span": 0,
			"tokens_dropped_missing_hidden": 0,
			"candidate_matches_by_index": 0,
			"candidate_matches_by_fingerprint": 0,
		}

		self._build_feature_index()
		self._build_line_items()

		if not self._items:
			raise RuntimeError("No usable line-level samples found.")
		if self.layers is None:
			raise RuntimeError("Unable to resolve hidden-state layers from matched samples.")
		if self.hidden_size is None:
			raise RuntimeError("Unable to infer hidden size from matched samples.")
		if self.feature_dim is None:
			raise RuntimeError("Unable to infer feature dimension from matched samples.")

		LOGGER.info(
			"Loaded line dataset from %s with %s samples (rows=%s, matched_candidates=%s, unmatched_candidates=%s)",
			self.jsonl_path,
			len(self._items),
			self.stats["jsonl_rows_total"],
			self.stats["jsonl_candidates_matched"],
			self.stats["jsonl_candidates_unmatched"],
		)

	def __len__(self) -> int:
		return len(self._items)

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
		item = self._items[index]
		payload = self._load_payload(item.path)
		feature_dict = payload.get("features", {})
		candidate = self._lookup_candidate(feature_dict, item.candidate_key)
		if candidate is None:
			raise KeyError(f"Candidate {item.candidate_key} not found in {item.path}")
		hidden_states = candidate.get("hidden_states")
		if not isinstance(hidden_states, Mapping):
			raise KeyError(f"hidden_states missing for candidate {item.candidate_key} in {item.path}")

		vectors: List[torch.Tensor] = []
		assert self.layers is not None
		for layer in self.layers:
			layer_dict = self._lookup_layer(hidden_states, layer)
			if not isinstance(layer_dict, Mapping):
				raise KeyError(f"Layer {layer} missing for candidate {item.candidate_key} in {item.path}")
			if self.line_representation == "last_token":
				vector = self._lookup_vector(layer_dict, item.token_positions[-1])
				if vector is None:
					raise KeyError(
						f"Token position {item.token_positions[-1]} missing for layer {layer} in {item.path}"
					)
				vectors.append(vector.to(torch.float32))
			else:
				line_vectors: List[torch.Tensor] = []
				for token_position in item.token_positions:
					vector = self._lookup_vector(layer_dict, token_position)
					if vector is None:
						raise KeyError(
							f"Token position {token_position} missing for layer {layer} in {item.path}"
						)
					line_vectors.append(vector.to(torch.float32))
				vectors.append(self._aggregate_line_vectors(line_vectors))

		features = self._combine_vectors(vectors)
		label = torch.tensor(item.label, dtype=torch.float32)
		metadata = {
			"line_number": torch.tensor(item.line_number, dtype=torch.int64),
			"token_count": torch.tensor(len(item.token_positions), dtype=torch.int64),
		}
		return features, label, metadata

	def _build_feature_index(self) -> None:
		for split in self.feature_splits:
			split_dir = self.root / split
			if not split_dir.exists():
				LOGGER.warning("Feature split directory missing: %s", split_dir)
				continue
			for file_path in sorted(split_dir.glob("*.pt")):
				self.stats["feature_files_scanned"] += 1
				try:
					payload = torch.load(file_path, map_location="cpu")
				except Exception as exc:  # pylint: disable=broad-except
					LOGGER.warning("Skipping unreadable feature file %s: %s", file_path, exc)
					self.stats["feature_files_skipped"] += 1
					continue
				self.stats["feature_files_loaded"] += 1
				sample_id = self._extract_sample_id(payload, file_path)
				if sample_id is None:
					self.stats["feature_files_skipped"] += 1
					continue
				entry = FeatureEntry(path=file_path, split=split, sample_id=sample_id)
				self._feature_index.setdefault(sample_id, []).append(entry)

	def _build_line_items(self) -> None:
		for record in self._iter_jsonl(self.jsonl_path):
			self.stats["jsonl_rows_total"] += 1
			sample_id = str(record.get("id", ""))
			if not sample_id:
				self.stats["jsonl_candidates_invalid"] += 1
				continue
			entries = self._ordered_entries(sample_id)
			if not entries:
				self.stats["samples_missing_features"] += 1
				continue
			token_labels_raw = record.get("token_labels")
			if not isinstance(token_labels_raw, list):
				self.stats["jsonl_candidates_invalid"] += 1
				continue
			self.stats["jsonl_rows_usable"] += 1

			for json_candidate_idx, token_labels in enumerate(token_labels_raw):
				self.stats["jsonl_candidates_total"] += 1
				if not isinstance(token_labels, list):
					self.stats["jsonl_candidates_invalid"] += 1
					continue
				line_error_map = self._extract_record_candidate_line_error_map(record, json_candidate_idx)
				resolved = self._resolve_feature_candidate(record, json_candidate_idx, entries)
				if resolved is None:
					self._handle_unmatched(
						f"Unable to map sample_id={sample_id} candidate_idx={json_candidate_idx}"
					)
					continue
				entry, candidate_key, candidate = resolved
				if not isinstance(candidate, Mapping):
					self._handle_unmatched(
						f"Mapped candidate is malformed for sample_id={sample_id} candidate_idx={json_candidate_idx}"
					)
					continue
				span = self._parse_code_range(candidate.get("code_token_idx"))
				if span is None:
					self._handle_unmatched(
						f"Missing/invalid code_token_idx for sample_id={sample_id} candidate_idx={json_candidate_idx}"
					)
					continue
				code_start, code_end = span
				hidden_states = candidate.get("hidden_states")
				if not isinstance(hidden_states, Mapping):
					self._handle_unmatched(
						f"hidden_states missing for sample_id={sample_id} candidate_idx={json_candidate_idx}"
					)
					continue
				candidate_layers = self._resolve_layers(hidden_states, candidate)
				if candidate_layers is None:
					self._handle_unmatched(
						f"Could not resolve requested layers for sample_id={sample_id} candidate_idx={json_candidate_idx}"
					)
					continue

				line_token_positions: Dict[int, List[int]] = {}
				line_token_correctness: Dict[int, List[bool]] = {}
				for token_offset, token_info in enumerate(token_labels):
					if not isinstance(token_info, Mapping):
						continue
					if self.drop_ws_comment and bool(token_info.get("ws_comment", False)):
						self.stats["tokens_dropped_ws_comment"] += 1
						continue
					line_number = self._extract_token_line_number(token_info)
					if line_number < 0:
						self.stats["tokens_dropped_no_line"] += 1
						continue
					token_position = code_start + token_offset
					if token_position > code_end:
						self.stats["tokens_dropped_out_of_code_span"] += 1
						continue
					if not self._has_required_vectors(hidden_states, candidate_layers, token_position):
						self.stats["tokens_dropped_missing_hidden"] += 1
						continue
					line_token_positions.setdefault(line_number, []).append(token_position)
					if "label" in token_info and token_info.get("label") is not None:
						line_token_correctness.setdefault(line_number, []).append(bool(token_info.get("label")))

				candidate_kept = 0
				for line_number, positions in line_token_positions.items():
					self.stats["line_labels_total"] += 1
					if not positions:
						continue
					line_label = self._lookup_line_label(line_error_map, line_number)
					if line_label < 0:
						line_label = self._derive_line_label_from_tokens(line_token_correctness.get(line_number, []))
					if line_label < 0:
						self.stats["lines_dropped_missing_label"] += 1
						continue

					if self.hidden_size is None:
						self.hidden_size = self._infer_hidden_size(hidden_states, candidate_layers, positions[0])
						if self.hidden_size is not None:
							self.feature_dim = (
								self.hidden_size * len(candidate_layers)
								if self.combiner == "concat"
								else self.hidden_size
							)
					self._items.append(
						LineSampleIndex(
							path=entry.path,
							candidate_key=candidate_key,
							line_number=line_number,
							label=float(line_label),
							token_positions=tuple(positions),
						)
					)
					self.stats["line_labels_kept"] += 1
					candidate_kept += 1

				if candidate_kept > 0:
					self.stats["jsonl_candidates_matched"] += 1
				else:
					self._handle_unmatched(
						f"No usable lines after filtering for sample_id={sample_id} candidate_idx={json_candidate_idx}"
					)

	def _aggregate_line_vectors(self, vectors: Sequence[torch.Tensor]) -> torch.Tensor:
		if not vectors:
			raise RuntimeError("No vectors to aggregate for line representation")
		if len(vectors) == 1:
			return vectors[0]
		stacked = torch.stack(list(vectors), dim=0)
		if self.aggregation == "max":
			return torch.max(stacked, dim=0).values
		return torch.mean(stacked, dim=0)

	def _derive_line_label_from_tokens(self, token_correctness: Sequence[bool]) -> int:
		if not token_correctness:
			return -1
		return 1 if any(not item for item in token_correctness) else 0

	def _ordered_entries(self, sample_id: str) -> List[FeatureEntry]:
		entries = list(self._feature_index.get(sample_id, []))
		if not entries:
			return []
		if len(entries) > 1:
			self.stats["feature_collisions"] += len(entries) - 1
			if self.warn_on_collision and sample_id not in self._collision_warned:
				self._collision_warned.add(sample_id)
				LOGGER.warning("Multiple feature files found for sample_id=%s (%s files)", sample_id, len(entries))
		priority = {name: idx for idx, name in enumerate(self._split_priority)}
		expected = self.expected_jsonl_split
		entries.sort(
			key=lambda entry: (
				0 if expected is not None and entry.split == expected else 1,
				priority.get(entry.split, len(priority) + 1),
				str(entry.path),
			)
		)
		return entries

	def _resolve_feature_candidate(
		self,
		record: Mapping[str, Any],
		json_candidate_idx: int,
		entries: Sequence[FeatureEntry],
	) -> Optional[Tuple[FeatureEntry, Hashable, Mapping[str, Any]]]:
		candidate_token_hash = self._extract_record_candidate_hash(record, json_candidate_idx)
		for entry in entries:
			payload = self._load_payload(entry.path)
			feature_dict = payload.get("features", {})
			if not isinstance(feature_dict, Mapping):
				continue

			if self.match_mode in {"auto", "index"}:
				candidate_key, candidate = self._candidate_by_index(feature_dict, json_candidate_idx)
				if candidate is not None:
					if candidate_token_hash is None or self._candidate_hash_matches(candidate, candidate_token_hash):
						self.stats["candidate_matches_by_index"] += 1
						return entry, candidate_key, candidate

			if self.match_mode in {"auto", "fingerprint"} and candidate_token_hash is not None:
				matched_key = self._candidate_key_by_hash(entry.path, payload, candidate_token_hash)
				if matched_key is not None:
					candidate = self._lookup_candidate(feature_dict, matched_key)
					if isinstance(candidate, Mapping):
						self.stats["candidate_matches_by_fingerprint"] += 1
						return entry, matched_key, candidate
		return None

	def _candidate_by_index(
		self,
		feature_dict: Mapping[Any, Any],
		json_candidate_idx: int,
	) -> Tuple[Hashable, Optional[Mapping[str, Any]]]:
		if json_candidate_idx in feature_dict:
			candidate = feature_dict[json_candidate_idx]
			return json_candidate_idx, candidate if isinstance(candidate, Mapping) else None
		str_key = str(json_candidate_idx)
		if str_key in feature_dict:
			candidate = feature_dict[str_key]
			return str_key, candidate if isinstance(candidate, Mapping) else None
		return json_candidate_idx, None

	def _candidate_key_by_hash(
		self,
		path: Path,
		payload: Mapping[str, Any],
		token_hash: str,
	) -> Optional[Hashable]:
		mapping = self._token_hash_cache.get(path)
		if mapping is None:
			mapping = self._build_candidate_hash_map(payload)
			self._token_hash_cache[path] = mapping
		keys = mapping.get(token_hash, [])
		if not keys:
			return None
		return sorted(keys, key=lambda item: str(item))[0]

	def _build_candidate_hash_map(self, payload: Mapping[str, Any]) -> Dict[str, List[Hashable]]:
		feature_dict = payload.get("features", {})
		if not isinstance(feature_dict, Mapping):
			return {}
		mapping: Dict[str, List[Hashable]] = {}
		for candidate_key, candidate in feature_dict.items():
			if not isinstance(candidate, Mapping):
				continue
			token_hash = self._token_hash_from_candidate(candidate)
			if token_hash is None:
				continue
			mapping.setdefault(token_hash, []).append(candidate_key)
		return mapping

	def _token_hash_from_candidate(self, candidate: Mapping[str, Any]) -> Optional[str]:
		token_ids = candidate.get("token_ids")
		if token_ids is None:
			return None
		normalized = self._normalize_token_ids(token_ids)
		if normalized is None:
			return None
		return self._stable_hash_int_list(normalized)

	def _extract_record_candidate_hash(self, record: Mapping[str, Any], candidate_idx: int) -> Optional[str]:
		token_ids = record.get("token_ids")
		if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
			return None
		if candidate_idx >= len(token_ids):
			return None
		candidate_token_ids = token_ids[candidate_idx]
		normalized = self._normalize_token_ids(candidate_token_ids)
		if normalized is None:
			return None
		return self._stable_hash_int_list(normalized)

	def _extract_record_candidate_line_error_map(
		self,
		record: Mapping[str, Any],
		candidate_idx: int,
	) -> Mapping[str, Any]:
		line_error_maps = record.get("line_error_map")
		if not isinstance(line_error_maps, Sequence) or isinstance(line_error_maps, (str, bytes)):
			return {}
		if candidate_idx >= len(line_error_maps):
			return {}
		candidate_map = line_error_maps[candidate_idx]
		if not isinstance(candidate_map, Mapping):
			return {}
		return candidate_map

	def _extract_token_line_number(self, token_info: Mapping[str, Any]) -> int:
		line_no = token_info.get("line")
		if line_no is None:
			return -1
		try:
			return int(line_no)
		except (TypeError, ValueError):
			return -1

	def _lookup_line_label(self, line_error_map: Mapping[str, Any], line_number: int) -> int:
		if line_number < 0:
			return -1
		line_key = str(line_number)
		if line_key not in line_error_map:
			return -1
		return 0 if bool(line_error_map[line_key]) else 1

	def _candidate_hash_matches(self, candidate: Mapping[str, Any], token_hash: str) -> bool:
		candidate_hash = self._token_hash_from_candidate(candidate)
		if candidate_hash is None:
			return True
		return candidate_hash == token_hash

	def _parse_code_range(self, value: Any) -> Optional[Tuple[int, int]]:
		if not isinstance(value, (list, tuple)) or len(value) < 2:
			return None
		try:
			start = int(value[0])
			end = int(value[1])
		except (TypeError, ValueError):
			return None
		if start < 0 or end < start:
			return None
		return start, end

	def _resolve_layers(
		self,
		hidden_states: Mapping[Any, Any],
		candidate: Mapping[str, Any],
	) -> Optional[Tuple[int, ...]]:
		available_layers = self._extract_layer_keys(hidden_states)
		if not available_layers:
			return None
		if self.layers is None:
			payload_layers = candidate.get("hidden_state_layers")
			resolved = self._extract_layer_sequence(payload_layers)
			if resolved is None:
				resolved = available_layers
			resolved = tuple(layer for layer in resolved if layer in available_layers)
			if not resolved:
				resolved = available_layers
			self.layers = resolved
			self.layer_count = len(self.layers)
		assert self.layers is not None
		for layer in self.layers:
			if layer not in available_layers:
				return None
		return self.layers

	def _extract_layer_keys(self, hidden_states: Mapping[Any, Any]) -> Tuple[int, ...]:
		layers: List[int] = []
		for key in hidden_states.keys():
			try:
				layers.append(int(key))
			except (TypeError, ValueError):
				continue
		if not layers:
			return ()
		return tuple(sorted(set(layers)))

	def _extract_layer_sequence(self, value: Any) -> Optional[Tuple[int, ...]]:
		if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
			return None
		layers: List[int] = []
		for item in value:
			try:
				layers.append(int(item))
			except (TypeError, ValueError):
				return None
		if not layers:
			return None
		return tuple(layers)

	def _has_required_vectors(
		self,
		hidden_states: Mapping[Any, Any],
		layers: Sequence[int],
		token_position: int,
	) -> bool:
		for layer in layers:
			layer_dict = self._lookup_layer(hidden_states, layer)
			if not isinstance(layer_dict, Mapping):
				return False
			if self._lookup_vector(layer_dict, token_position) is None:
				return False
		return True

	def _infer_hidden_size(
		self,
		hidden_states: Mapping[Any, Any],
		layers: Sequence[int],
		token_position: int,
	) -> Optional[int]:
		for layer in layers:
			layer_dict = self._lookup_layer(hidden_states, layer)
			if not isinstance(layer_dict, Mapping):
				continue
			vector = self._lookup_vector(layer_dict, token_position)
			if isinstance(vector, torch.Tensor):
				return int(vector.numel())
		return None

	def _lookup_layer(self, hidden_states: Mapping[Any, Any], layer: int) -> Optional[Mapping[Any, Any]]:
		if layer in hidden_states and isinstance(hidden_states[layer], Mapping):
			return hidden_states[layer]
		str_key = str(layer)
		layer_dict = hidden_states.get(str_key)
		return layer_dict if isinstance(layer_dict, Mapping) else None

	def _lookup_vector(self, layer_dict: Mapping[Any, Any], token_position: int) -> Optional[torch.Tensor]:
		if token_position in layer_dict and isinstance(layer_dict[token_position], torch.Tensor):
			return layer_dict[token_position]
		str_key = str(token_position)
		vector = layer_dict.get(str_key)
		return vector if isinstance(vector, torch.Tensor) else None

	def _combine_vectors(self, vectors: Sequence[torch.Tensor]) -> torch.Tensor:
		vec_list = list(vectors)
		if not vec_list:
			raise RuntimeError("No vectors provided for feature combination.")
		if self.combiner == "stack":
			return torch.stack(vec_list, dim=0)
		if len(vec_list) == 1:
			return vec_list[0]
		return torch.cat(vec_list, dim=0)

	def _load_payload(self, path: Path) -> Dict[str, Any]:
		if self._payload_cache_path == path and self._payload_cache is not None:
			return self._payload_cache
		payload: Dict[str, Any] = torch.load(path, map_location="cpu")
		self._payload_cache_path = path
		self._payload_cache = payload
		return payload

	def _extract_sample_id(self, payload: Mapping[str, Any], file_path: Path) -> Optional[str]:
		sample_id_raw = payload.get("sample_id")
		if sample_id_raw is not None:
			sample_id = str(sample_id_raw)
			if sample_id:
				return sample_id
		name = file_path.stem
		if "__" in name:
			prefix = name.split("__", 1)[0]
			if prefix:
				return prefix
		return None

	def _lookup_candidate(self, feature_dict: Mapping[Any, Any], candidate_key: Hashable) -> Optional[Mapping[str, Any]]:
		if candidate_key in feature_dict:
			candidate = feature_dict[candidate_key]
			return candidate if isinstance(candidate, Mapping) else None
		if isinstance(candidate_key, int):
			alt_key = str(candidate_key)
		else:
			try:
				alt_key = int(str(candidate_key))
			except (TypeError, ValueError):
				alt_key = None
		if alt_key is not None and alt_key in feature_dict:
			candidate = feature_dict[alt_key]
			return candidate if isinstance(candidate, Mapping) else None
		return None

	def _handle_unmatched(self, message: str) -> None:
		if self.strict_mapping:
			raise RuntimeError(message)
		self.stats["jsonl_candidates_unmatched"] += 1
		LOGGER.debug(message)

	@staticmethod
	def _canonical_split_name(value: Optional[str]) -> str:
		raw = (value or "").strip().lower()
		if raw in {"val", "valid", "validation"}:
			return "validation"
		if raw in {"test", "testing"}:
			return "test"
		if raw in {"train", "training"}:
			return "train"
		return raw

	@staticmethod
	def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
		with path.open("r", encoding="utf-8") as handle:
			for line in handle:
				line = line.strip()
				if not line:
					continue
				yield json.loads(line)

	@staticmethod
	def _normalize_token_ids(value: Any) -> Optional[List[int]]:
		if isinstance(value, torch.Tensor):
			if value.ndim == 0:
				return [int(value.item())]
			return [int(v) for v in value.tolist()]
		if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
			return None
		result: List[int] = []
		for item in value:
			try:
				result.append(int(item))
			except (TypeError, ValueError):
				return None
		return result

	@staticmethod
	def _stable_hash_int_list(values: Sequence[int]) -> str:
		payload = json.dumps(list(values), separators=(",", ":")).encode("utf-8")
		return hashlib.sha1(payload).hexdigest()
