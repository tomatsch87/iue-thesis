"""
Sequence dataset for dynamic token selection models.

Default dataset behavior:
- Reads `<root>/<split>/*.pt`, aligns candidate code spans, and returns
	`(sequence_tensor, label)` per candidate.
- If `layers` is omitted, layers are inferred from payload metadata/keys.
- By default, empty code spans are filtered (`filter_empty=True` in training configs).
- Sequence entries keep per-token per-layer vectors with shape
	`[num_tokens, num_layers, hidden_size]` before collate-time flattening.

Additional options:
- `token_filter` supports: `new-line`, `top_k-token-entropy`, `top_k-logtoku`,
	`lines+top_k-logtoku`, `new-line+top_k-token-entropy`,
	`comment-or-identifier`, `comment-or-identifier+top_k-token-entropy`.
- `token_filter_config` controls filter behavior (for example `k`, `quantile`,
	thresholds, and `add_last_token` in top-k modes).
- `tokenizer_name`, injected `tokenizer`, and injected `parser` allow external control
	over token decoding and tree-sitter parsing for structural filters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

@dataclass(frozen=True)
class SampleIndex:
	path: Path
	candidate_idx: int
	label: float
	start_pos: int
	end_pos: int
	prompt_offset: int


@dataclass(frozen=True)
class TokenSelection:
	rel_indices: List[int]
	abs_positions: List[int]


_IM_END_TOKEN = "<|im_end|>"

TokenFilter = Callable[[torch.Tensor, Mapping[str, Any], int, int, int], torch.Tensor]


class HiddenStateSequenceDataset(Dataset):
	def __init__(
		self,
		root: Path,
		split: str,
		layers: Optional[Sequence[int]] = None,
		filter_empty: bool = True,
		token_filter: Optional[str] = None,
		token_filter_config: Optional[Mapping[str, Any]] = None,
		tokenizer_name: Optional[str] = None,
		tokenizer: Optional[Any] = None,
		parser: Optional[Any] = None,
	) -> None:
		self.root = root
		self.split = split
		self.filter_empty = filter_empty
		self.layers: Optional[Tuple[int, ...]] = tuple(layers) if layers is not None else None
		self.layer_count: Optional[int] = None if self.layers is None else len(self.layers)
		self.hidden_size: Optional[int] = None
		self._items: List[SampleIndex] = []
		self._tokenizer = tokenizer
		self._python_parser = parser
		self._token_filter_name = token_filter
		self._token_filter_fn: Optional[TokenFilter] = None
		self._token_filter_config: Dict[str, Any] = dict(token_filter_config or {})
		self._tokenizer_name = tokenizer_name or DEFAULT_MODEL_NAME
		self._cache_path: Optional[Path] = None
		self._cache_payload: Optional[Dict[str, Any]] = None

		self._token_filter_registry: Dict[str, TokenFilter] = {
			"new-line": self._filter_new_line_tokens,
			"top_k-token-entropy": self._filter_topk_token_entropy,
			"top_k-logtoku": self._filter_topk_logtoku,
			"lines+top_k-logtoku": self._filter_lines_topk_logtoku,
			"new-line+top_k-token-entropy": self._filter_new_line_then_topk,
			"comment-or-identifier": self._filter_comment_identifier_tokens,
			"comment-or-identifier+top_k-token-entropy": self._filter_comment_identifier_then_topk,
		}
		LOGGER.info("Selected token filter: %s", self._token_filter_name or "none")
		LOGGER.info("Token filter config: %s", self._token_filter_config)

		split_dir = self.root / self.split
		if not split_dir.exists():
			raise FileNotFoundError(f"Split directory not found: {split_dir}")

		files = sorted(split_dir.glob("*.pt"))
		if not files:
			raise FileNotFoundError(f"No .pt files found in {split_dir}")

		skipped_no_label = 0
		skipped_missing_sequence = 0
		skipped_missing_layers = 0

		for file_path in files:
			payload: Dict[str, Any] = torch.load(file_path, map_location="cpu")
			feature_dict = payload.get("features", {})
			if not isinstance(feature_dict, dict):
				continue
			prompt_offset = self._safe_int(payload.get("prompt_token_count"))
			if prompt_offset is None or prompt_offset < 0:
				prompt_offset = 0
			payload_layers = self._sanitize_layer_list(payload.get("hidden_state_layers"))
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
					skipped_missing_layers += 1
					continue
				candidate_layers = self._resolve_layers(payload_layers, hidden_states)
				if candidate_layers is None:
					skipped_missing_layers += 1
					continue
				if self.layers is None:
					self.layers = candidate_layers
					self.layer_count = len(self.layers)
				elif tuple(candidate_layers) != self.layers:
					missing = set(self.layers) - set(candidate_layers)
					if missing:
						skipped_missing_layers += 1
						continue
				aligned_end = self._candidate_has_required_tokens(
					candidate,
					hidden_states,
					start_pos,
					end_pos,
				)
				if aligned_end is None:
					skipped_missing_sequence += 1
					continue
				if self.hidden_size is None:
					self.hidden_size = self._infer_hidden_size(hidden_states)
				self._items.append(
					SampleIndex(
						path=file_path,
						candidate_idx=int(cand_idx),
						label=float(label),
						start_pos=start_pos,
						end_pos=aligned_end,
						prompt_offset=prompt_offset,
					)
				)

		if not self._items:
			raise RuntimeError("No usable samples found; verify that hidden states include the requested sequence.")
		if self.layers is None:
			raise RuntimeError("Unable to determine layer configuration from the provided features.")
		if self.hidden_size is None:
			raise RuntimeError("Unable to infer hidden size from the provided features.")

		if self._token_filter_name is not None:
			self._initialize_token_filter(self._token_filter_name)

		if skipped_no_label or skipped_missing_sequence or skipped_missing_layers:
			LOGGER.info(
				"Loaded split %s with %s samples (skipped %s without labels, %s without sequences, %s without layers)",
				split,
				len(self._items),
				skipped_no_label,
				skipped_missing_sequence,
				skipped_missing_layers,
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
		sequence = self._build_sequence(hidden_states, item.start_pos, item.end_pos, context=item.path.name)
		if self._token_filter_fn is not None:
			sequence = self._token_filter_fn(
				sequence,
				candidate,
				item.start_pos,
				item.end_pos,
				item.prompt_offset,
			)
		label = torch.tensor(item.label, dtype=torch.float32)
		return sequence, label

	def _initialize_token_filter(self, filter_name: str) -> None:
		filter_name = filter_name.lower()
		if filter_name not in self._token_filter_registry:
			raise ValueError(f"Unsupported token filter '{filter_name}'.")
		if self._tokenizer is None:
			self._tokenizer = self._load_tokenizer()
		if self._python_parser is None:
			self._python_parser = self._load_python_parser()
		self._token_filter_fn = self._token_filter_registry[filter_name]

	def _build_sequence(
		self,
		hidden_states: Mapping[Any, Any],
		start_pos: int,
		end_pos: int,
		*,
		context: Optional[str] = None,
	) -> torch.Tensor:
		if self.layers is None or self.layer_count is None:
			raise RuntimeError("Layer configuration not initialized.")
		positions = range(start_pos, end_pos + 1)
		per_token: List[torch.Tensor] = []
		for pos in positions:
			layer_vectors: List[torch.Tensor] = []
			for layer in self.layers:
				layer_dict = hidden_states.get(layer)
				if not isinstance(layer_dict, Mapping):
					raise KeyError(f"Layer {layer} missing for sequence construction{self._format_context(context)}")
				vector = self._get_vector(layer_dict, pos)
				if vector is None:
					raise KeyError(
						f"Token position {pos} missing for layer {layer}{self._format_context(context)}"
					)
				layer_vectors.append(vector.to(torch.float32))
			per_token.append(torch.stack(layer_vectors, dim=0))
		return torch.stack(per_token, dim=0)

	def _candidate_has_required_tokens(
		self,
		candidate: Mapping[str, Any],
		hidden_states: Mapping[Any, Any],
		start_pos: int,
		end_pos: int,
	) -> Optional[int]:
		aligned_end = self._locate_im_end_token(candidate)
		if aligned_end is None:
			return None
		if aligned_end != end_pos:
			if aligned_end < start_pos:
				return None
			LOGGER.debug(
				"Adjusting end position from %s to %s to align with %s token",
				end_pos,
				aligned_end,
				_IM_END_TOKEN,
			)
		layers = self.layers
		if layers is None:
			layers = self._resolve_layers(None, hidden_states)
		if layers is None:
			return None
		for layer in layers:
			layer_dict = hidden_states.get(layer)
			if not isinstance(layer_dict, Mapping):
				return None
			for idx in range(start_pos, aligned_end + 1):
				if self._get_vector(layer_dict, idx) is None:
					return None
		return aligned_end

	def _filter_new_line_tokens(
		self,
		sequence: torch.Tensor,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> torch.Tensor:
		selection, fallback = self._compute_newline_selection(candidate, start_pos, end_pos, prompt_offset)
		if selection is None:
			return self._apply_filter_fallback(sequence, fallback)
		index_tensor = torch.as_tensor(selection.rel_indices, dtype=torch.long, device=sequence.device)
		return sequence.index_select(0, index_tensor)

	def _filter_topk_token_entropy(
		self,
		sequence: torch.Tensor,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> torch.Tensor:
		_ = prompt_offset  # unused but matches interface
		selected_positions = self._select_topk_entropy_positions(
			candidate,
			start_pos,
			end_pos,
		)
		if not selected_positions:
			return self._first_last_sequence(sequence)
		rel_indices = [pos - start_pos for pos in selected_positions if start_pos <= pos <= end_pos]
		if not rel_indices:
			return self._first_last_sequence(sequence)
		index_tensor = torch.as_tensor(rel_indices, dtype=torch.long, device=sequence.device)
		filtered = sequence.index_select(0, index_tensor)
		return self._append_last_token_if_configured(filtered, sequence)

	def _filter_topk_logtoku(
		self,
		sequence: torch.Tensor,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> torch.Tensor:
		_ = prompt_offset  # unused but matches interface
		selected_positions = self._select_topk_logtoku_positions(
			candidate,
			start_pos,
			end_pos,
		)
		if not selected_positions:
			return self._first_last_sequence(sequence)
		rel_indices = [pos - start_pos for pos in selected_positions if start_pos <= pos <= end_pos]
		if not rel_indices:
			return self._first_last_sequence(sequence)
		index_tensor = torch.as_tensor(rel_indices, dtype=torch.long, device=sequence.device)
		filtered = sequence.index_select(0, index_tensor)
		return self._append_last_token_if_configured(filtered, sequence)

	def _filter_lines_topk_logtoku(
		self,
		sequence: torch.Tensor,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> torch.Tensor:
		rel_positions = self._select_linewise_topk_logtoku_positions(
			candidate,
			start_pos,
			end_pos,
			prompt_offset,
		)
		if not rel_positions:
			return self._first_last_sequence(sequence)
		index_tensor = torch.as_tensor(rel_positions, dtype=torch.long, device=sequence.device)
		filtered = sequence.index_select(0, index_tensor)
		return self._append_last_token_if_configured(filtered, sequence)

	def _filter_new_line_then_topk(
		self,
		sequence: torch.Tensor,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> torch.Tensor:
		selection, fallback = self._compute_newline_selection(candidate, start_pos, end_pos, prompt_offset)
		if selection is None:
			intermediate = self._apply_filter_fallback(sequence, fallback)
			positions = self._positions_for_fallback(intermediate.size(0), start_pos, end_pos, fallback)
		else:
			index_tensor = torch.as_tensor(selection.rel_indices, dtype=torch.long, device=sequence.device)
			intermediate = sequence.index_select(0, index_tensor)
			positions = selection.abs_positions
		if not positions:
			return self._first_last_sequence(intermediate)
		selected_positions = self._select_topk_entropy_positions(
			candidate,
			start_pos,
			end_pos,
			allowed_positions=positions,
		)
		if not selected_positions:
			return self._first_last_sequence(intermediate)
		index_lookup = {pos: idx for idx, pos in enumerate(positions)}
		rel_indices = [index_lookup[pos] for pos in selected_positions if pos in index_lookup]
		if not rel_indices:
			return self._first_last_sequence(intermediate)
		index_tensor = torch.as_tensor(rel_indices, dtype=torch.long, device=intermediate.device)
		filtered = intermediate.index_select(0, index_tensor)
		return self._append_last_token_if_configured(filtered, sequence)

	def _filter_comment_identifier_then_topk(
		self,
		sequence: torch.Tensor,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> torch.Tensor:
		selection, _ = self._compute_node_type_selection(
			candidate,
			start_pos,
			end_pos,
			prompt_offset,
			target_node_types={"comment", "identifier"},
		)
		if selection is None:
			fallback_mode = "original"
			intermediate = self._apply_filter_fallback(sequence, fallback_mode)
			positions = self._positions_for_fallback(intermediate.size(0), start_pos, end_pos, fallback_mode)
		else:
			excluded = set(selection.abs_positions)
			positions = [pos for pos in range(start_pos, end_pos + 1) if pos not in excluded]
			if not positions:
				intermediate = self._first_last_sequence(sequence)
				positions = self._positions_for_fallback(
					intermediate.size(0),
					start_pos,
					end_pos,
					"first_last",
				)
			else:
				rel_indices = [pos - start_pos for pos in positions]
				if not rel_indices:
					return self._first_last_sequence(sequence)
				index_tensor = torch.as_tensor(rel_indices, dtype=torch.long, device=sequence.device)
				intermediate = sequence.index_select(0, index_tensor)
		if not positions:
			return self._first_last_sequence(intermediate)
		selected_positions = self._select_topk_entropy_positions(
			candidate,
			start_pos,
			end_pos,
			allowed_positions=positions,
		)
		if not selected_positions:
			return self._first_last_sequence(intermediate)
		index_lookup = {pos: idx for idx, pos in enumerate(positions)}
		rel_indices = [index_lookup[pos] for pos in selected_positions if pos in index_lookup]
		if not rel_indices:
			return self._first_last_sequence(intermediate)
		index_tensor = torch.as_tensor(rel_indices, dtype=torch.long, device=intermediate.device)
		filtered = intermediate.index_select(0, index_tensor)
		return self._append_last_token_if_configured(filtered, sequence)

	def _append_last_token_if_configured(
		self, filtered: torch.Tensor, original_sequence: torch.Tensor
	) -> torch.Tensor:
		if not self._should_append_last_token():
			return filtered
		if original_sequence.size(0) == 0:
			return filtered
		last_vec = original_sequence[-1:].to(filtered.device)
		return torch.cat([filtered, last_vec], dim=0)

	def _should_append_last_token(self) -> bool:
		config = self._token_filter_config or {}
		if not bool(config.get("add_last_token")):
			return False
		k_val = self._safe_int(config.get("k"))
		return k_val is not None and k_val > 0

	def _filter_comment_identifier_tokens(
		self,
		sequence: torch.Tensor,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> torch.Tensor:
		selection, _ = self._compute_node_type_selection(
			candidate,
			start_pos,
			end_pos,
			prompt_offset,
			target_node_types={"comment", "identifier"},
		)
		if selection is None:
			return self._apply_filter_fallback(sequence, "original")
		excluded = set(selection.abs_positions)
		kept_positions = [pos for pos in range(start_pos, end_pos + 1) if pos not in excluded]
		if not kept_positions:
			return self._first_last_sequence(sequence)
		rel_indices = [pos - start_pos for pos in kept_positions]
		if not rel_indices:
			return self._first_last_sequence(sequence)
		index_tensor = torch.as_tensor(rel_indices, dtype=torch.long, device=sequence.device)
		return sequence.index_select(0, index_tensor)

	@staticmethod
	def _first_last_sequence(sequence: torch.Tensor) -> torch.Tensor:
		length = sequence.size(0)
		if length <= 2:
			return sequence
		indices = torch.tensor([0, length - 1], dtype=torch.long, device=sequence.device)
		return sequence.index_select(0, indices)

	def _apply_filter_fallback(self, sequence: torch.Tensor, fallback: Optional[str]) -> torch.Tensor:
		if fallback == "original":
			return sequence
		return self._first_last_sequence(sequence)

	def _positions_for_fallback(
		self,
		sequence_length: int,
		start_pos: int,
		end_pos: int,
		fallback: Optional[str],
	) -> List[int]:
		if sequence_length <= 0:
			return []
		if fallback == "original":
			return [start_pos + idx for idx in range(sequence_length)]
		if fallback == "first_last":
			if sequence_length <= 2:
				return [start_pos + idx for idx in range(sequence_length)]
			return [start_pos, end_pos]
		return []

	def _compute_newline_selection(
		self,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> Tuple[Optional[TokenSelection], Optional[str]]:
		if self._tokenizer is None or self._python_parser is None:
			return None, "first_last"
		token_data = self._extract_token_slice(candidate, start_pos, end_pos, prompt_offset)
		if token_data is None:
			return None, "first_last"
		token_ids, positions = token_data
		if not token_ids:
			return None, "first_last"
		pieces, spans, positions = self._tokens_to_text(token_ids, positions)
		text = "".join(pieces)
		if not text.strip():
			return None, "first_last"
		try:
			tree = self._python_parser.parse(text.encode("utf-8"))
		except Exception as exc:  # pylint: disable=broad-except
			LOGGER.warning("Failed to parse python snippet for token filtering: %s", exc)
			return None, "first_last"
		target_positions = self._locate_newline_token_positions(tree, spans, positions, start_pos)
		if start_pos <= end_pos and end_pos not in target_positions:
			target_positions.append(end_pos)
		target_positions = sorted(target_positions)
		if not target_positions:
			return None, "first_last"
		abs_positions = [pos for pos in target_positions if start_pos <= pos <= end_pos]
		if not abs_positions:
			return None, "first_last"
		rel_indices = [pos - start_pos for pos in abs_positions]
		if not rel_indices:
			return None, "first_last"
		return TokenSelection(rel_indices=rel_indices, abs_positions=abs_positions), None

	def _compute_node_type_selection(
		self,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
		*,
		target_node_types: Set[str],
	) -> Tuple[Optional[TokenSelection], Optional[str]]:
		if self._tokenizer is None or self._python_parser is None:
			return None, "first_last"
		if not target_node_types:
			return None, "first_last"
		token_data = self._extract_token_slice(candidate, start_pos, end_pos, prompt_offset)
		if token_data is None:
			return None, "first_last"
		token_ids, positions = token_data
		if not token_ids:
			return None, "first_last"
		pieces, spans, positions = self._tokens_to_text(token_ids, positions)
		text = "".join(pieces)
		if not text.strip():
			return None, "first_last"
		try:
			tree = self._python_parser.parse(text.encode("utf-8"))
		except Exception as exc:  # pylint: disable=broad-except
			LOGGER.warning("Failed to parse python snippet for token filtering: %s", exc)
			return None, "first_last"
		target_positions = self._locate_tokens_by_node_types(
			tree,
			spans,
			positions,
			start_pos,
			target_node_types,
		)
		if not target_positions:
			return None, "first_last"
		abs_positions = [pos for pos in target_positions if start_pos <= pos <= end_pos]
		if not abs_positions:
			return None, "first_last"
		rel_indices = [pos - start_pos for pos in abs_positions]
		if not rel_indices:
			return None, "first_last"
		return TokenSelection(rel_indices=rel_indices, abs_positions=abs_positions), None

	def _select_topk_entropy_positions(
		self,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		*,
		allowed_positions: Optional[Sequence[int]] = None,
	) -> Optional[List[int]]:
		entropies = candidate.get("token_entropies")
		if not isinstance(entropies, Mapping):
			return None
		config = self._token_filter_config if isinstance(self._token_filter_config, Mapping) else {}
		k_override = self._safe_int(config.get("k"))
		use_k_override = k_override is not None and k_override > 0
		if use_k_override:
			quantile = 0.05
		else:
			quantile = self._coerce_float(config.get("quantile"))
			if quantile is None or not (0.0 < quantile <= 1.0):
				quantile = 0.05
		min_entropy = None if use_k_override else self._coerce_float(config.get("min_entropy"))
		max_entropy = None if use_k_override else self._coerce_float(config.get("max_entropy"))
		normalize = False if use_k_override else bool(config.get("normalize_entropy"))
		norm_divisor: Optional[float] = None
		if normalize:
			if self._tokenizer is None:
				self._tokenizer = self._load_tokenizer()
			tokenizer = self._tokenizer
			vocab_size: Optional[int] = None
			if tokenizer is not None:
				vocab_size = getattr(tokenizer, "vocab_size", None)
				if vocab_size is None and hasattr(tokenizer, "get_vocab"):
					try:
						vocab_size = len(tokenizer.get_vocab())  # type: ignore[arg-type]
					except Exception:  # pylint: disable=broad-except
						vocab_size = None
			if vocab_size is None or vocab_size <= 1:
				normalize = False
			else:
				norm = math.log(vocab_size)
				if norm <= 0:
					normalize = False
				else:
					norm_divisor = norm
		allowed_set: Optional[Set[int]] = set(allowed_positions) if allowed_positions is not None else None
		available: List[Tuple[int, float]] = []
		for rel_pos_raw, entropy_val in entropies.items():
			abs_pos = self._safe_int(rel_pos_raw)
			if abs_pos is None or abs_pos < start_pos or abs_pos > end_pos:
				continue
			if allowed_set is not None and abs_pos not in allowed_set:
				continue
			try:
				entropy_float = float(entropy_val)
			except (TypeError, ValueError):
				continue
			if normalize and norm_divisor is not None:
				entropy_float = entropy_float / norm_divisor
				if entropy_float < 0.0:
					entropy_float = 0.0
				elif entropy_float > 1.0:
					entropy_float = 1.0
			available.append((abs_pos, entropy_float))
		if not available:
			return None
		filtered = [
			(pos, entropy)
			for pos, entropy in available
			if (min_entropy is None or entropy >= min_entropy)
			and (max_entropy is None or entropy <= max_entropy)
		]
		if not filtered:
			return None
		total = len(filtered)
		if use_k_override:
			top_k = min(k_override, total)  # type: ignore[arg-type]
		else:
			top_k = max(1, round(int(total * quantile)))
			top_k = min(top_k, total)
		sorted_candidates = sorted(filtered, key=lambda item: (-item[1], item[0]))
		selected_positions = sorted(pos for pos, _ in sorted_candidates[:top_k])
		return selected_positions

	def _select_topk_logtoku_positions(
		self,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		*,
		allowed_positions: Optional[Sequence[int]] = None,
	) -> Optional[List[int]]:
		reliabilities = candidate.get("logtoku_reliability")
		if not isinstance(reliabilities, Mapping):
			return None
		config = self._token_filter_config if isinstance(self._token_filter_config, Mapping) else {}
		k_override = self._safe_int(config.get("k"))
		use_k_override = k_override is not None and k_override > 0
		if use_k_override:
			quantile = 0.05
		else:
			quantile = self._coerce_float(config.get("quantile"))
			if quantile is None or not (0.0 < quantile <= 1.0):
				quantile = 0.05
		min_rel = None if use_k_override else self._coerce_float(config.get("min_reliability"))
		max_rel = None if use_k_override else self._coerce_float(config.get("max_reliability"))
		allowed_set: Optional[Set[int]] = set(allowed_positions) if allowed_positions is not None else None
		available: List[Tuple[int, float]] = []
		for rel_pos_raw, reliability_val in reliabilities.items():
			abs_pos = self._safe_int(rel_pos_raw)
			if abs_pos is None or abs_pos < start_pos or abs_pos > end_pos:
				continue
			if allowed_set is not None and abs_pos not in allowed_set:
				continue
			try:
				reliability_float = float(reliability_val)
			except (TypeError, ValueError):
				continue
			available.append((abs_pos, reliability_float))
		if not available:
			return None
		filtered = [
			(pos, reliability)
			for pos, reliability in available
			if (min_rel is None or reliability >= min_rel)
			and (max_rel is None or reliability <= max_rel)
		]
		if not filtered:
			return None
		total = len(filtered)
		if use_k_override:
			top_k = min(k_override, total)  # type: ignore[arg-type]
		else:
			top_k = max(1, round(int(total * quantile)))
			top_k = min(top_k, total)
		sorted_candidates = sorted(filtered, key=lambda item: (item[1], item[0]))
		selected_positions = sorted(pos for pos, _ in sorted_candidates[:top_k])
		return selected_positions

	def _select_linewise_topk_logtoku_positions(
		self,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> Optional[List[int]]:
		if self._tokenizer is None:
			self._tokenizer = self._load_tokenizer()
		if self._python_parser is None:
			self._python_parser = self._load_python_parser()
		if self._tokenizer is None or self._python_parser is None:
			return None
		token_data = self._extract_token_slice(candidate, start_pos, end_pos, prompt_offset)
		if token_data is None:
			return None
		token_ids, positions = token_data
		if not token_ids:
			return None
		pieces, spans, positions = self._tokens_to_text(token_ids, positions)
		text = "".join(pieces)
		reliabilities = candidate.get("logtoku_reliability")
		if not isinstance(reliabilities, Mapping):
			return None
		k_override = self._safe_int(self._token_filter_config.get("k")) if isinstance(self._token_filter_config, Mapping) else None
		k_per_line = k_override if k_override is not None and k_override > 0 else 1
		line_lookup: Dict[int, List[int]] = {}
		for idx, (span_start, _) in enumerate(spans):
			line_num = text.count("\n", 0, span_start)
			line_lookup.setdefault(line_num, []).append(idx)
		selected_positions: List[int] = []
		for line_tokens in line_lookup.values():
			candidates: List[Tuple[int, float]] = []
			for token_idx in line_tokens:
				if token_idx >= len(positions):
					continue
				abs_pos = positions[token_idx]
				if abs_pos < start_pos or abs_pos > end_pos:
					continue
				rel_val = reliabilities.get(abs_pos)
				if rel_val is None:
					rel_val = reliabilities.get(str(abs_pos))
				if rel_val is None:
					continue
				try:
					rel_float = float(rel_val)
				except (TypeError, ValueError):
					continue
				candidates.append((abs_pos, rel_float))
			if not candidates:
				continue
			candidates_sorted = sorted(candidates, key=lambda item: (item[1], item[0]))
			for abs_pos, _ in candidates_sorted[:k_per_line]:
				selected_positions.append(abs_pos)
		if not selected_positions:
			return None
		selected_positions = sorted(set(selected_positions))
		rel_indices = [pos - start_pos for pos in selected_positions if start_pos <= pos <= end_pos]
		return rel_indices if rel_indices else None

	def _extract_token_slice(
		self,
		candidate: Mapping[str, Any],
		start_pos: int,
		end_pos: int,
		prompt_offset: int,
	) -> Optional[Tuple[List[int], List[int]]]:
		token_ids = candidate.get("token_ids")
		if token_ids is None:
			return None
		if isinstance(token_ids, torch.Tensor):
			flat_ids = token_ids.tolist()
		elif isinstance(token_ids, Sequence):
			flat_ids = [int(x) for x in token_ids]
		else:
			return None
		total = len(flat_ids)
		assistant_start = start_pos + prompt_offset
		if total == 0 or assistant_start >= total:
			LOGGER.warning(
				"token_ids missing assistant positions [%s, %s) (prompt_offset=%s, total=%s)",
				start_pos,
				end_pos,
				prompt_offset,
				total,
			)
			return None
		assistant_end = min(end_pos + prompt_offset, total - 1)
		if assistant_end < assistant_start:
			return None
		slice_ids = flat_ids[assistant_start : assistant_end + 1]
		positions = list(range(start_pos, start_pos + len(slice_ids)))
		return slice_ids, positions

	def _tokens_to_text(
		self,
		token_ids: Sequence[int],
		positions: Sequence[int],
	) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
		pieces: List[str] = []
		spans: List[Tuple[int, int]] = []
		pos_copy: List[int] = []
		cursor = 0
		for token_id, pos in zip(token_ids, positions):
			piece = self._decode_token(token_id)
			encoded = piece.encode("utf-8")
			length = len(encoded)
			spans.append((cursor, cursor + length))
			cursor += length
			pieces.append(piece)
			pos_copy.append(pos)
		return pieces, spans, pos_copy

	def _decode_token(self, token_id: int) -> str:
		if self._tokenizer is None:
			return ""
		try:
			token_text = self._tokenizer.convert_ids_to_tokens([token_id])[0]
		except Exception:  # pylint: disable=broad-except
			token_text = None
		decoded = self._tokenizer.decode(
			[token_id],
			skip_special_tokens=False,
			clean_up_tokenization_spaces=False,
		)
		decoded = decoded.replace("Ċ", "\n")
		special_tokens = {"```", "python", _IM_END_TOKEN}
		if token_text in special_tokens or decoded in special_tokens:
			return ""
		return decoded

	def _locate_newline_token_positions(
		self,
		tree: Any,
		spans: Sequence[Tuple[int, int]],
		positions: Sequence[int],
		start_pos: int,
	) -> List[int]:
		line_best: Dict[int, Tuple[int, int]] = {}
		for node in self._iter_leaf_nodes(tree.root_node):
			if getattr(node, "type", None) == "comment":
				continue
			token_idx = self._token_index_for_byte(node.start_byte, spans)
			if token_idx is None:
				continue
			row, col = node.start_point
			current = line_best.get(row)
			if current is None or col < current[0]:
				line_best[row] = (col, token_idx)
		if not line_best:
			return []
		selected: List[int] = []
		for _, token_idx in sorted(line_best.values(), key=lambda item: positions[item[1]]):
			abs_pos = positions[token_idx]
			if abs_pos < start_pos:
				continue
			selected.append(abs_pos)
		return selected

	def _locate_tokens_by_node_types(
		self,
		tree: Any,
		spans: Sequence[Tuple[int, int]],
		positions: Sequence[int],
		start_pos: int,
		target_node_types: Set[str],
	) -> List[int]:
		if not target_node_types:
			return []
		selected: Set[int] = set()
		for node in self._iter_leaf_nodes(tree.root_node):
			node_type = getattr(node, "type", None)
			if node_type not in target_node_types:
				continue
			start_byte = getattr(node, "start_byte", None)
			end_byte = getattr(node, "end_byte", None)
			if start_byte is None or end_byte is None:
				continue
			indices = self._token_indices_for_span(start_byte, end_byte, spans)
			if not indices:
				continue
			for token_idx in indices:
				if token_idx >= len(positions):
					continue
				abs_pos = positions[token_idx]
				if abs_pos < start_pos:
					continue
				selected.add(abs_pos)
		return sorted(selected)

	@staticmethod
	def _iter_leaf_nodes(node: Any):  # type: ignore[override]
		if getattr(node, "child_count", 0) == 0:
			yield node
			return
		for child in getattr(node, "children", []) or []:
			yield from HiddenStateSequenceDataset._iter_leaf_nodes(child)

	@staticmethod
	def _token_index_for_byte(byte_pos: int, spans: Sequence[Tuple[int, int]]) -> Optional[int]:
		for idx, (start, end) in enumerate(spans):
			if start <= byte_pos < end:
				return idx
		return None

	@staticmethod
	def _token_indices_for_span(
		start_byte: int,
		end_byte: int,
		spans: Sequence[Tuple[int, int]],
	) -> List[int]:
		indices: List[int] = []
		for idx, (span_start, span_end) in enumerate(spans):
			if span_end <= start_byte:
				continue
			if span_start >= end_byte:
				break
			indices.append(idx)
		return indices

	def _locate_im_end_token(self, candidate: Mapping[str, Any]) -> Optional[int]:
		resolved = candidate.get("resolved_hidden_state_tokens")
		if not isinstance(resolved, Sequence):
			return None
		for entry in resolved:
			if not isinstance(entry, (list, tuple)) or len(entry) < 2:
				continue
			pos_raw, token_text = entry[0], entry[1]
			if not isinstance(token_text, str):
				continue
			pos_val = self._safe_int(pos_raw)
			if pos_val is None:
				continue
			if token_text == _IM_END_TOKEN:
				return pos_val
		return None

	def _infer_hidden_size(self, hidden_states: Mapping[Any, Any]) -> int:
		layers = self.layers
		if layers is None:
			raise RuntimeError("Cannot infer hidden size before layers are determined")
		for layer in layers:
			layer_dict = hidden_states.get(layer)
			if not isinstance(layer_dict, Mapping):
				continue
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

	def _resolve_layers(
		self,
		payload_layers: Optional[Tuple[int, ...]],
		hidden_states: Mapping[Any, Any],
	) -> Optional[Tuple[int, ...]]:
		if payload_layers:
			return payload_layers
		layer_keys = self._sanitize_layer_list(hidden_states.keys())
		return layer_keys

	def _sanitize_layer_list(
		self,
		candidates: Optional[Iterable[Any]],
	) -> Optional[Tuple[int, ...]]:
		if candidates is None:
			return None
		resolved: List[int] = []
		for layer in candidates:
			try:
				resolved.append(int(layer))
			except (TypeError, ValueError):
				return None
		if not resolved:
			return None
		return tuple(sorted(resolved))

	def _get_vector(self, layer_dict: Mapping[Any, Any], token_idx: int) -> Optional[torch.Tensor]:
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

	def _load_tokenizer(self):
		try:
			from transformers import AutoTokenizer  # type: ignore
		except ImportError as exc:  # pragma: no cover
			raise ImportError("Token filtering requires the 'transformers' package.") from exc
		return AutoTokenizer.from_pretrained(self._tokenizer_name)

	@staticmethod
	def _load_python_parser():
		try:
			from tree_sitter_language_pack import get_parser  # type: ignore
		except ImportError as exc:  # pragma: no cover
			raise ImportError("Token filtering requires the 'tree_sitter_language_pack' package.") from exc
		return get_parser("python")

	@staticmethod
	def _format_context(context: Optional[str]) -> str:
		return f" in {context}" if context else ""

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

	@staticmethod
	def _coerce_float(value: Any) -> Optional[float]:
		if value is None:
			return None
		try:
			return float(value)
		except (TypeError, ValueError):
			return None

