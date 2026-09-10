import shutil
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import torch

from .dataset import HiddenStateSequenceDataset

IM_END = "<|im_end|>"
REAL_SAMPLE_PATH = (Path(__file__).resolve().parent.parent / "data/feats_lcb_qwen3_code_segment/train/1873_B__hsL48__hsTcode_to_end__attLnone__attTnone.pt").resolve()


def _make_hidden_state(value: float, size: int = 2) -> torch.Tensor:
	return torch.tensor([value] * size, dtype=torch.float32)


def _resolved_tokens(total: int, marker_idx: int) -> list[tuple[int, str]]:
	tokens: list[tuple[int, str]] = []
	for idx in range(total):
		token = IM_END if idx == marker_idx else f"tok_{idx}"
		if idx == marker_idx + 1:
			token = "Ċ"
		tokens.append((idx, token))
	return tokens


def _prepare_real_sample_tmpdir(tmpdir: str) -> Path:
	if not REAL_SAMPLE_PATH.exists():
		pytest.skip(f"Real sample not found at {REAL_SAMPLE_PATH}")
	root = Path(tmpdir)
	split_dir = root / "train"
	split_dir.mkdir(parents=True, exist_ok=True)
	dest = split_dir / REAL_SAMPLE_PATH.name
	shutil.copy(REAL_SAMPLE_PATH, dest)
	return root


def test_sequence_dataset_returns_full_code_to_penultimate():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		payload = {
			"hidden_state_layers": [0],
			"features": {
				0: {
					"is_correct": 1.0,
					"assistant_token_count": 6,
					"code_token_idx": (1, 4),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(0, 6)},
					},
					"resolved_hidden_state_tokens": _resolved_tokens(6, 4),
				},
			}
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(root=root, split="train")

		assert len(dataset) == 1
		assert dataset.layer_count == 1
		assert dataset.hidden_size == 2

		sequence, label = dataset[0]
		assert label.item() == 1.0
		assert sequence.shape == (4, 1, 2)
		# expect first token (pos=1) to contain layer 0 -> [1,1]
		expected_first = torch.tensor([[1.0, 1.0]])
		assert torch.allclose(sequence[0], expected_first)
		# expect final token (pos=4) to contain layer 0 -> [4,4]
		expected_last = torch.tensor([[4.0, 4.0]])
		assert torch.allclose(sequence[-1], expected_last)


def test_sequence_dataset_skips_invalid_candidates():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		payload = {
			"hidden_state_layers": [0],
			"features": {
				0: {  # invalid: penultimate < code start
					"is_correct": 0.0,
					"assistant_token_count": 4,
					"code_token_idx": (3, 3),
					"hidden_states": {0: {3: _make_hidden_state(3.0)}},
					"resolved_hidden_state_tokens": _resolved_tokens(4, 2),
				},
				1: {
					"is_correct": 1.0,
					"assistant_token_count": 5,
					"code_token_idx": (1, 3),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(1, 5)},
					},
					"resolved_hidden_state_tokens": _resolved_tokens(5, 3),
				},
			},
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(root=root, split="train")

		assert len(dataset) == 1
		sequence, label = dataset[0]
		assert label.item() == 1.0
		# assistant_token_count=5 -> penultimate index=3, start=1 => positions 1..3
		assert sequence.shape == (3, 1, 2)
		expected_tokens = torch.tensor(
			[
				[[1.0, 1.0]],
				[[2.0, 2.0]],
				[[3.0, 3.0]],
			]
		)
		assert torch.allclose(sequence, expected_tokens)


def test_sequence_dataset_respects_requested_layers():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		payload = {
			"hidden_state_layers": [0, 1],
			"features": {
				0: {
					"is_correct": 0.0,
					"assistant_token_count": 5,
					"code_token_idx": (0, 2),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(0, 5)},
						1: {idx: _make_hidden_state(float(idx + 5)) for idx in range(0, 5)},
					},
					"resolved_hidden_state_tokens": _resolved_tokens(5, 3),
				},
			},
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(root=root, split="train", layers=[1])

		assert dataset.layer_count == 1
		sequence, _ = dataset[0]
		# assistant_token_count=5 -> penultimate index=3, start=0 => positions 0..3
		assert sequence.shape == (4, 1, 2)
		# Only layer 1 values should be present
		expected = torch.tensor(
			[
				[[5.0, 5.0]],
				[[6.0, 6.0]],
				[[7.0, 7.0]],
				[[8.0, 8.0]],
			]
		)
		assert torch.allclose(sequence, expected)


def test_sequence_dataset_aligns_end_with_im_end_token():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		payload = {
			"hidden_state_layers": [0],
			"features": {
				0: {
					"is_correct": 1.0,
					"assistant_token_count": 8,
					"code_token_idx": (1, 6),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(0, 8)},
					},
					"resolved_hidden_state_tokens": _resolved_tokens(8, 5),
				},
			},
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(root=root, split="train")

		sequence, _ = dataset[0]
		# code_start=1, im_end located at 5 => indices 1..5
		assert sequence.shape == (5, 1, 2)
		assert torch.allclose(sequence[-1], torch.tensor([[5.0, 5.0]]))


def test_sequence_dataset_real_sample_default(tmp_path_factory: pytest.TempPathFactory):
	temp_dir = tmp_path_factory.mktemp("real_sample_default")
	root = _prepare_real_sample_tmpdir(str(temp_dir))
	dataset = HiddenStateSequenceDataset(root=root, split="train")

	assert len(dataset) == 10
	sequence, label = dataset[0]
	assert sequence.shape == (130, 1, 2048)
	assert label == 1.0
	assert sequence[-1][0][0].item() == 0.57421875


def test_sequence_dataset_real_sample_new_line_filter(tmp_path_factory: pytest.TempPathFactory):
	temp_dir = tmp_path_factory.mktemp("real_sample_newline")
	root = _prepare_real_sample_tmpdir(str(temp_dir))
	dataset = HiddenStateSequenceDataset(
		root=root,
		split="train",
		token_filter="new-line",
	)

	assert len(dataset) == 10
	sequence, label = dataset[1]
	assert sequence.shape == (16, 1, 2048)
	assert label == 1.0
	assert sequence[-2][0][0].item() == -2.515625


def test_sequence_dataset_real_sample_comment_identifier_filter(tmp_path_factory: pytest.TempPathFactory):
	temp_dir = tmp_path_factory.mktemp("real_sample_comment_identifier")
	root = _prepare_real_sample_tmpdir(str(temp_dir))
	dataset = HiddenStateSequenceDataset(
		root=root,
		split="train",
		token_filter="comment-or-identifier",
	)

	assert len(dataset) == 10
	sequence, label = dataset[2]
	assert sequence.shape == (67, 1, 2048)
	assert label == 1.0
	assert sequence[-1][0][0].item() == 0.99609375


def test_sequence_dataset_topk_token_entropy_filter():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		token_entropies: Dict[int, float] = {idx: float(idx) / 10.0 for idx in range(41)}
		token_entropies[5] = 100.0
		token_entropies[10] = 70.0
		token_entropies[17] = 90.0
		token_entropies[23] = 80.0

		payload = {
			"hidden_state_layers": [0],
			"prompt_token_count": 0,
			"features": {
				0: {
					"is_correct": 1.0,
					"assistant_token_count": 42,
					"code_token_idx": (0, 40),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(41)},
					},
					"token_entropies": token_entropies,
					"resolved_hidden_state_tokens": _resolved_tokens(42, 40),
				},
			},
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(root=root, split="train", token_filter="top_k-token-entropy")

		sequence, label = dataset[0]
		assert label.item() == 1.0
		assert sequence.shape == (2, 1, 2)
		selected_positions = sequence[:, 0, 0].tolist()
		assert selected_positions == [5.0, 17.0]


def test_sequence_dataset_topk_token_entropy_filter_explicit_k():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		token_entropies: Dict[int, float] = {
			0: 0.1,
			1: 0.5,
			2: 0.2,
			3: 0.9,
			4: 0.7,
			5: 0.6,
		}

		payload = {
			"hidden_state_layers": [0],
			"prompt_token_count": 0,
			"features": {
				0: {
					"is_correct": 1.0,
					"assistant_token_count": 7,
					"code_token_idx": (0, 5),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(6)},
					},
					"token_entropies": token_entropies,
					"resolved_hidden_state_tokens": _resolved_tokens(7, 5),
				},
			},
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(
			root=root,
			split="train",
			token_filter="top_k-token-entropy",
			token_filter_config={"k": 3, "min_entropy": 1.0, "quantile": 0.9},
		)

		sequence, label = dataset[0]
		assert label.item() == 1.0
		assert sequence.shape == (3, 1, 2)
		selected_positions = sequence[:, 0, 0].tolist()
		assert selected_positions == [3.0, 4.0, 5.0]


def test_sequence_dataset_topk_token_entropy_filter_custom_quantile():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		token_entropies: Dict[int, float] = {idx: float(idx) for idx in range(22)}

		payload = {
			"hidden_state_layers": [0],
			"prompt_token_count": 0,
			"features": {
				0: {
					"is_correct": 1.0,
					"assistant_token_count": 23,
					"code_token_idx": (0, 21),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(22)},
					},
					"token_entropies": token_entropies,
					"resolved_hidden_state_tokens": _resolved_tokens(23, 21),
				},
			},
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(
			root=root,
			split="train",
			token_filter="top_k-token-entropy",
			token_filter_config={"quantile": 0.2},
		)

		sequence, _ = dataset[0]
		assert sequence.shape == (4, 1, 2)
		selected_positions = sequence[:, 0, 0].tolist()
		assert selected_positions == [18.0, 19.0, 20.0, 21.0]


def test_sequence_dataset_topk_token_entropy_filter_entropy_range():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		token_entropies: Dict[int, float] = {idx: float(idx) for idx in range(18)}

		payload = {
			"hidden_state_layers": [0],
			"prompt_token_count": 0,
			"features": {
				0: {
					"is_correct": 1.0,
					"assistant_token_count": 20,
					"code_token_idx": (0, 18),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(19)},
					},
					"token_entropies": token_entropies,
					"resolved_hidden_state_tokens": _resolved_tokens(20, 18),
				},
			},
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(
			root=root,
			split="train",
			token_filter="top_k-token-entropy",
			token_filter_config={"min_entropy": 5.5, "max_entropy": 7.1},
		)

		sequence, _ = dataset[0]
		assert sequence.shape == (1, 1, 2)
		selected_positions = sequence[:, 0, 0].tolist()
		assert selected_positions == [7.0]


def test_sequence_dataset_topk_token_entropy_filter_appends_last_token():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		token_entropies: Dict[int, float] = {
			0: 0.1,
			1: 0.9,
			2: 0.2,
			3: 0.8,
			4: 0.05,
		}

		payload = {
			"hidden_state_layers": [0],
			"prompt_token_count": 0,
			"features": {
				0: {
					"is_correct": 1.0,
					"assistant_token_count": 6,
					"code_token_idx": (0, 4),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(5)},
					},
					"token_entropies": token_entropies,
					"resolved_hidden_state_tokens": _resolved_tokens(6, 4),
				},
			},
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(
			root=root,
			split="train",
			token_filter="top_k-token-entropy",
			token_filter_config={"k": 2, "add_last_token": True},
		)

		sequence, label = dataset[0]
		assert label.item() == 1.0
		# top-2 entropies -> positions 1 and 3, then append last token (pos=4)
		assert sequence.shape == (3, 1, 2)
		selected_positions = sequence[:, 0, 0].tolist()
		assert selected_positions == [1.0, 3.0, 4.0]


def test_sequence_dataset_topk_token_entropy_filter_appends_last_token_even_if_selected():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		# Make the last token (pos=4) part of the top-k selection already.
		token_entropies: Dict[int, float] = {
			0: 0.1,
			1: 0.8,
			2: 0.2,
			3: 0.3,
			4: 0.9,
		}

		payload = {
			"hidden_state_layers": [0],
			"prompt_token_count": 0,
			"features": {
				0: {
					"is_correct": 1.0,
					"assistant_token_count": 6,
					"code_token_idx": (0, 4),
					"hidden_states": {
						0: {idx: _make_hidden_state(float(idx)) for idx in range(5)},
					},
					"token_entropies": token_entropies,
					"resolved_hidden_state_tokens": _resolved_tokens(6, 4),
				},
			},
		}
		torch.save(payload, split_dir / "sample.pt")

		dataset = HiddenStateSequenceDataset(
			root=root,
			split="train",
			token_filter="top_k-token-entropy",
			token_filter_config={"k": 2, "add_last_token": True},
		)

		sequence, label = dataset[0]
		assert label.item() == 1.0
		# top-2 entropies -> positions 1 and 4, then append last token (pos=4) again => K+1 length
		assert sequence.shape == (3, 1, 2)
		selected_positions = sequence[:, 0, 0].tolist()
		assert selected_positions == [1.0, 4.0, 4.0]