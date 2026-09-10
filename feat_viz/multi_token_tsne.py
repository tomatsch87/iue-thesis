from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
PROJECT_ROOT = PACKAGE_PARENT.parent
for candidate in (PACKAGE_PARENT, PROJECT_ROOT):
	if str(candidate) not in sys.path:
		sys.path.insert(0, str(candidate))

from baselines.dataset import HiddenStateDataset


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Visualize hidden state embeddings aggregated over multiple layers/token positions",
	)
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
		"--layers",
		type=int,
		nargs="+",
		default=[48],
		help="One or more hidden-state layer indices to concatenate",
	)
	parser.add_argument(
		"--token-positions",
		type=int,
		nargs="+",
		default=[0],
		help=(
			"Assistant token positions to extract for each layer (supports negative indices as offsets "
			"from the final assistant tokens)"
		),
	)
	parser.add_argument(
		"--combiner",
		type=str,
		default="concat",
		help="How to combine layer/token vectors; matches HiddenStateDataset options (concat or average)",
	)
	parser.add_argument(
		"--filter-empty",
		action="store_true",
		help="Skip candidates whose code span is empty",
	)
	parser.add_argument(
		"--max-samples",
		type=int,
		default=None,
		help="Optional cap on number of dataset samples to visualize",
	)
	parser.add_argument(
		"--color-by-difficulty",
		action="store_true",
		help="Color marker borders by sample difficulty",
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
		help="Random state for reproducibility",
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
		default=Path("multi_token_tsne.png"),
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


def _resolve_difficulty(dataset: HiddenStateDataset, index: int) -> str:
	# HiddenStateDataset already caches payloads, so reuse its helpers for metadata.
	sample_meta = dataset._items[index]  # type: ignore[attr-defined]
	payload = dataset._load_payload(sample_meta.path)  # type: ignore[attr-defined]
	value = payload.get("difficulty")
	if value is None or value == "":
		return "unknown"
	return str(value)


def collect_embeddings(
	dataset: HiddenStateDataset,
	max_samples: int | None,
	rng_seed: int,
) -> Tuple[np.ndarray, List[str], List[str]]:
	total = len(dataset)
	if total == 0:
		raise ValueError("Dataset is empty")
	indices = np.arange(total)
	if max_samples is not None:
		if max_samples <= 0:
			raise ValueError("max-samples must be positive")
		if max_samples < total:
			rng = np.random.default_rng(rng_seed)
			indices = rng.choice(indices, size=max_samples, replace=False)

	embeddings: List[np.ndarray] = []
	labels: List[str] = []
	difficulties: List[str] = []

	for idx in indices.tolist():
		feature_tensor, label_tensor = dataset[idx]
		embeddings.append(feature_tensor.detach().cpu().numpy())
		label_value = float(label_tensor.item())
		if np.isclose(label_value, 1.0):
			labels.append("correct")
		elif np.isclose(label_value, 0.0):
			labels.append("incorrect")
		else:
			labels.append("unknown")
		difficulties.append(_resolve_difficulty(dataset, idx))

	if not embeddings:
		raise ValueError("No embeddings extracted from dataset")

	embeddings_array = np.stack(embeddings, axis=0)
	embeddings_array = normalize(embeddings_array, axis=1)
	return embeddings_array, labels, difficulties


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
	difficulties: Iterable[str],
	output_path: Path,
	use_difficulty_colors: bool,
) -> None:
	label_array = np.array(list(labels))
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
	difficulty_colors = {
		"easy": "tab:blue",
		"medium": "tab:orange",
		"hard": "tab:pink",
	}

	plt.figure(figsize=(8, 6))
	ax = plt.gca()
	for label in unique_labels:
		mask = label_array == label
		if use_difficulty_colors and difficulty_array is not None:
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
			color=colors.get(label, "tab:blue"),
			edgecolors=edge_colors,
			linewidths=line_width,
		)

	plt.title("t-SNE of Hidden States (multi-token)")
	base_legend = ax.legend(title="Correctness", loc="upper right")
	ax.add_artist(base_legend)

	if use_difficulty_colors and difficulty_array is not None:
		difficulties_present = sorted({str(value) for value in difficulty_array})
		if difficulties_present:
			handles: List[Line2D] = []
			for name in difficulties_present:
				handles.append(
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
			if handles:
				labels_out = [str(handle.get_label()) for handle in handles]
				ax.legend(handles, labels_out, title="Difficulty", loc="lower right")

	plt.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=300)
	LOGGER.info("Saved t-SNE plot to %s", output_path)


def main() -> None:
	args = parse_args()
	setup_logging(args.log_level)

	data_root = args.data_dir.expanduser().resolve()
	output_path = args.output.expanduser().resolve()

	dataset = HiddenStateDataset(
		root=data_root,
		split=args.split,
		layers=args.layers,
		token_positions=args.token_positions,
		filter_empty=args.filter_empty,
		combiner=args.combiner,
	)

	embeddings, labels, difficulties = collect_embeddings(
		dataset,
		max_samples=args.max_samples,
		rng_seed=args.random_state,
	)
	LOGGER.info("Prepared %s embeddings", embeddings.shape[0])

	coordinates = run_tsne(
		embeddings,
		perplexity=args.perplexity,
		random_state=args.random_state,
		n_iter=args.n_iter,
	)
	use_difficulty_colors = bool(args.color_by_difficulty)
	plot_tsne(
		coordinates,
		labels,
		difficulties,
		output_path,
		use_difficulty_colors,
	)


if __name__ == "__main__":
	main()