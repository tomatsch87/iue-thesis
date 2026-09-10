from __future__ import annotations

import argparse
import logging
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Iterable, List, Tuple

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize


LOGGER = logging.getLogger(__name__)


def token_position_specifier(value: str) -> str | int:
	normalized = value.strip().lower()
	if normalized == "all":
		return "all"
	try:
		return int(value)
	except ValueError as exc:
		raise argparse.ArgumentTypeError(
			"Token position must be an integer index or 'all'"
		) from exc


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize hidden state embeddings with t-SNE")
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path("../data/feats_lcb_qwen3"),
		help="Directory containing extracted feature .pt files",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="train",
		help="Dataset split directory to load (e.g. train, validation)",
	)
	parser.add_argument(
		"--layer",
		type=int,
		default=48,
		help="Hidden state layer index to visualize",
	)
	parser.add_argument(
		"--token-position",
		type=token_position_specifier,
		default=token_position_specifier("0"),
		help=(
			"Relative assistant token position to visualize (e.g. 0, -1, all, -2/-3 "
			"when exactly four positions are present)"
		),
	)
	parser.add_argument(
		"--color-by-difficulty",
		action="store_true",
		help=(
			"Color marker borders by sample difficulty when token position is not 'all'. "
			"Defaults to disabled to preserve previous behavior."
		),
	)
	parser.add_argument(
		"--perplexity",
		type=float,
		default=40.0,
		help="t-SNE perplexity; will be clipped to valid range given data size",
	)
	parser.add_argument(
		"--random-state",
		type=int,
		default=42,
		help="Random state for t-SNE reproducibility",
	)
	parser.add_argument(
		"--n-iter",
		type=int,
		default=2000,
		help="Number of t-SNE iterations",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("tsne_layer48_token0.png"),
		help="Path to save the generated plot",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default="INFO",
		help="Logging level",
	)
	return parser.parse_args()


def setup_logging(level: str) -> None:
	logging.basicConfig(
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
		level=getattr(logging, level.upper(), logging.INFO),
	)



def select_token_tensors(
	layer_store: object,
	token_position: str | int,
) -> List[Tuple[int, torch.Tensor]]:
	positions: List[Tuple[int, torch.Tensor]] = []
	if isinstance(layer_store, Mapping):
		iterator = layer_store.items()
	elif isinstance(layer_store, Sequence) and not isinstance(layer_store, (str, bytes, bytearray)):
		iterator = enumerate(layer_store)
	else:
		return []

	for raw_index, tensor in iterator:
		try:
			index = int(raw_index)
		except (TypeError, ValueError):
			continue
		if tensor is None or not hasattr(tensor, "detach"):
			continue
		positions.append((index, tensor))

	if not positions:
		return []

	positions.sort(key=lambda pair: pair[0])

	if token_position == "all":
		return positions

	if isinstance(token_position, str):
		try:
			token_position = int(token_position)
		except ValueError:
			return []

	if isinstance(token_position, int) and token_position in (-2, -3):
		# Handle special cases for -2 and -3 when exactly three positions are present
		if len(positions) == 4 or len(positions) == 3:
			return [positions[token_position]]
		else:
			return []


	if isinstance(token_position, int) and token_position < 0:
		adjusted_index = len(positions) + token_position
		if adjusted_index < 0 or adjusted_index >= len(positions):
			return []
		return [positions[adjusted_index]]

	if isinstance(token_position, int):
		for index, tensor in positions:
			if index == token_position:
				return [(index, tensor)]
		return []

	return []



def collect_embeddings(
	split_path: Path,
	layer: int,
	token_position: str | int,
) -> Tuple[np.ndarray, List[str], List[str | None], List[str]]:
	embeddings: List[np.ndarray] = []
	labels: List[str] = []
	token_roles: List[str | None] = []
	difficulties: List[str] = []

	files: Iterable[Path] = sorted(split_path.glob("*.pt"))
	for file_path in files:
		payload = torch.load(file_path, map_location="cpu")
		difficulty = str(payload.get("difficulty", "")) or "unknown"
		feature_map = payload.get("features", {})
		for candidate in feature_map.values():
			hidden_states = candidate.get("hidden_states", {})
			layer_store = hidden_states.get(layer)
			if layer_store is None:
				continue
			selected_entries = select_token_tensors(layer_store, token_position)
			if not selected_entries:
				continue
			label_value = candidate.get("is_correct")
			if isinstance(label_value, bool):
				label = "correct" if label_value else "incorrect"
			elif label_value in (0, 1):
				label = "correct" if label_value == 1 else "incorrect"
			else:
				label = "unknown"
			role_map: dict[int, str] = {}
			if token_position == "all":
				entry_count = len(selected_entries)
				if entry_count not in (2, 3, 4):
					LOGGER.warning(
						"Skipping candidate with unexpected token count %s for 'all' specifier in %s",
						entry_count,
						file_path,
					)
					continue
				indices = [index for index, _ in selected_entries]
				if entry_count == 2:
					role_map[indices[0]] = "first token"
					role_map[indices[1]] = "last token"
				elif entry_count == 3:
					role_map[indices[0]] = "first code token"
					role_map[indices[1]] = "last code token"
					role_map[indices[2]] = "last token"
				else:
					role_map[indices[0]] = "first token"
					role_map[indices[1]] = "first code token"
					role_map[indices[2]] = "last code token"
					role_map[indices[3]] = "last token"
			for index, tensor in selected_entries:
				embeddings.append(tensor.detach().float().cpu().numpy())
				labels.append(label)
				token_roles.append(role_map.get(index) if token_position == "all" else None)
				difficulties.append(difficulty)

	if not embeddings:
		raise ValueError(
			f"No embeddings found for layer {layer} token spec {token_position} in {split_path}"
		)

	embeddings_array = np.stack(embeddings, axis=0)
	embeddings_array = normalize(embeddings_array, axis=1)
	return embeddings_array, labels, token_roles, difficulties


def run_tsne(
	embeddings: np.ndarray,
	perplexity: float,
	random_state: int,
	n_iter: int,
) -> np.ndarray:
	sample_count = embeddings.shape[0]
	if sample_count < 3:
		raise ValueError("Need at least three embeddings for t-SNE")

	max_perplexity = max(5.0, min(perplexity, (sample_count - 1) / 3))
	if max_perplexity != perplexity:
		LOGGER.info(
			"Adjusted perplexity from %.2f to %.2f for %s samples",
			perplexity,
			max_perplexity,
			sample_count,
		)

	tsne = TSNE(
		n_components=2,
		perplexity=max_perplexity,
		random_state=random_state,
		max_iter=n_iter,
		init="pca",
		learning_rate="auto",
		method="barnes_hut",
		metric="cosine",
	)
	LOGGER.info("Running t-SNE on %s embeddings", sample_count)
	return tsne.fit_transform(embeddings)


def plot_tsne(
	coordinates: np.ndarray,
	labels: Iterable[str],
	token_roles: Iterable[str | None],
	difficulties: Iterable[str],
	output_path: Path,
	use_token_roles: bool,
	use_difficulty_colors: bool,
) -> None:
	label_array = np.array(list(labels))
	role_array = np.array(list(token_roles), dtype=object) if use_token_roles else None
	difficulty_array = (
		np.array(list(difficulties), dtype=object)
		if use_difficulty_colors
		else None
	)
	unique_labels = sorted(set(label_array))
	colors = {
		"correct": "tab:green",
		"incorrect": "tab:red",
	}
	role_colors = {
		"first token": "tab:blue",
		"first code token": "tab:cyan",
		"last code token": "tab:orange",
		"last token": "tab:pink",
	}
	if difficulty_array is not None:
		difficulty_colors = {
			"easy": "tab:blue",
			"medium": "tab:orange",
			"hard": "tab:pink",
		}
	else:
		difficulty_colors = {}

	plt.figure(figsize=(8, 6))
	ax = plt.gca()
	for label in unique_labels:
		mask = label_array == label
		color_key = label
		if use_token_roles and role_array is not None:
			edge_colors = [role_colors.get(role, "none") for role in role_array[mask]]
			line_width = 0.8
			marker_size = 25
		elif use_difficulty_colors and difficulty_array is not None:
			edge_colors = [difficulty_colors.get(str(value), "none") for value in difficulty_array[mask]]
			line_width = 0.8
			marker_size = 25
		else:
			edge_colors = "none"
			line_width = 0.0
			marker_size = 30
		ax.scatter(
			coordinates[mask, 0],
			coordinates[mask, 1],
			s=marker_size,
			alpha=0.5,
			label=label,
			color=colors.get(color_key, colors.get(label, "tab:blue")),
			edgecolors=edge_colors,
			linewidths=line_width,
		)

	plt.title("t-SNE of Hidden States")
	base_legend = ax.legend(title="Correctness", loc="upper right")
	ax.add_artist(base_legend)

	if use_token_roles and role_array is not None:
		roles_present = sorted({role for role in role_array if role in role_colors})
		other_present = any(role is not None and role not in role_colors for role in role_array)

		if roles_present or other_present:
			role_handles: List[Line2D] = []
			for name in roles_present:
				role_handles.append(
					Line2D(
						[0],
						[0],
						marker="o",
						color="white",
						markerfacecolor="white",
						markeredgecolor=role_colors[name],
						markersize=5,
						linewidth=0.8,
						label=name,
					),
				)
			if other_present:
				role_handles.append(
					Line2D(
						[0],
						[0],
						marker="o",
						color="white",
						markerfacecolor="white",
						markeredgecolor="none",
						markersize=5,
						linewidth=0.8,
						label="other",
					),
				)
			if role_handles:
				role_labels = [str(handle.get_label()) for handle in role_handles]
				ax.legend(role_handles, role_labels, title="Token position", loc="lower right")
	elif use_difficulty_colors and difficulty_array is not None:
		difficulties_present = sorted({str(value) for value in difficulty_array})
		if difficulties_present:
			difficulty_handles: List[Line2D] = []
			for name in difficulties_present:
				difficulty_handles.append(
					Line2D(
						[0],
						[0],
						marker="o",
						color="white",
						markerfacecolor="white",
						markeredgecolor=difficulty_colors.get(name, "none"),
						markersize=5,
						linewidth=0.8,
						label=name,
					),
				)
			if difficulty_handles:
				difficulty_labels = [str(handle.get_label()) for handle in difficulty_handles]
				ax.legend(
					difficulty_handles,
					difficulty_labels,
					title="Difficulty",
					loc="lower right",
				)
	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300)
	LOGGER.info("Saved t-SNE plot to %s", output_path)


def main() -> None:
	args = parse_args()
	setup_logging(args.log_level)

	split_path = args.data_dir.expanduser().resolve() / args.split
	if not split_path.is_dir():
		raise FileNotFoundError(f"Split directory not found: {split_path}")

	embeddings, labels, token_roles, difficulties = collect_embeddings(
		split_path,
		args.layer,
		args.token_position,
	)
	LOGGER.info("Loaded %s embeddings", embeddings.shape[0])

	coordinates = run_tsne(embeddings, args.perplexity, args.random_state, args.n_iter)
	use_token_roles = isinstance(args.token_position, str) and args.token_position == "all"
	use_difficulty_colors = bool(args.color_by_difficulty and not use_token_roles)
	plot_tsne(
		coordinates,
		labels,
		token_roles,
		difficulties,
		args.output,
		use_token_roles,
		use_difficulty_colors,
	)


if __name__ == "__main__":
	main()