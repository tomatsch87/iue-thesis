import random
import tempfile
from pathlib import Path
import torch
from .dataset import HiddenStateDataset, HiddenStateRandomTailDataset

def test_hidden_state_dataset_basic():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()
		
		file_path = split_dir / "sample_0.pt"
		payload = {
			"features": {
				0: {
					"is_correct": 1.0,
					"hidden_states": {
						0: {0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])},
						1: {0: torch.tensor([5.0, 6.0]), 1: torch.tensor([7.0, 8.0])},
					}
				},
				1: {
					"is_correct": 0.0,
					"hidden_states": {
						0: {0: torch.tensor([9.0, 10.0]), 1: torch.tensor([11.0, 12.0])},
						1: {0: torch.tensor([13.0, 14.0]), 1: torch.tensor([15.0, 16.0])},
					}
				}
			}
		}
		torch.save(payload, file_path)
		
		dataset = HiddenStateDataset(root, "train", layers=[0, 1], token_positions=[0, 1])
		
		assert len(dataset) == 2
		assert dataset.feature_dim == 8  # 2 layers * 2 positions * 2 dims
		
		# Check first item
		features, label = dataset[0]
		assert features.shape == (8,)
		assert label.item() == 1.0
		expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
		assert torch.allclose(features, expected)
		
		# Check second item
		features, label = dataset[1]
		assert features.shape == (8,)
		assert label.item() == 0.0
		expected = torch.tensor([9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
		assert torch.allclose(features, expected)


def test_hidden_state_dataset_partial_tokens():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()
		
		file_path = split_dir / "sample_0.pt"
		payload = {
			"features": {
				0: {
					"is_correct": 1.0,
					"hidden_states": {
						0: {0: torch.tensor([1.0, 2.0]), 1: torch.tensor([3.0, 4.0])},
						1: {0: torch.tensor([5.0, 6.0]), 1: torch.tensor([7.0, 8.0])},
					}
				},
				1: {
					"is_correct": 0.0,
					"hidden_states": {
						0: {0: torch.tensor([9.0, 10.0]), 1: torch.tensor([11.0, 12.0])},
						1: {0: torch.tensor([13.0, 14.0]), 1: torch.tensor([15.0, 16.0])},
					}
				}
			}
		}
		torch.save(payload, file_path)
		
		dataset = HiddenStateDataset(root, "train", layers=[1], token_positions=[1])
		
		assert len(dataset) == 2
		assert dataset.feature_dim == 2  # 1 layer * 1 position * 2 dims

		# Check first item
		features, label = dataset[0]
		assert features.shape == (2,)
		assert label.item() == 1.0
		expected = torch.tensor([7.0, 8.0])
		assert torch.allclose(features, expected)

		# Check second item
		features, label = dataset[1]
		assert features.shape == (2,)
		assert label.item() == 0.0
		expected = torch.tensor([15.0, 16.0])
		assert torch.allclose(features, expected)


def test_hidden_state_dataset_negative_indexing():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()
		
		file_path = split_dir / "sample_0.pt"
		payload = {
			"features": {
				0: {
					"is_correct": 1.0,
					"hidden_states": {
						0: {0: torch.tensor([1.0]), 1: torch.tensor([2.0]), 2: torch.tensor([3.0])},
					}
				},
				1: {
					"is_correct": 0.0,
					"hidden_states": {
						0: {0: torch.tensor([4.0]), 1: torch.tensor([5.0]), 2: torch.tensor([6.0])},
					}
				}
			}
		}
		torch.save(payload, file_path)
		
		dataset = HiddenStateDataset(root, "train", layers=[0], token_positions=[-1])
		
		assert len(dataset) == 2
		assert dataset.feature_dim == 1  # 1 layer * 1 position * 1 dim

		# Check first item
		features, label = dataset[0]
		assert features.shape == (1,)
		assert label.item() == 1.0
		expected = torch.tensor([3.0])
		assert torch.allclose(features, expected)

		# Check second item
		features, label = dataset[1]
		assert features.shape == (1,)
		assert label.item() == 0.0
		expected = torch.tensor([6.0])
		assert torch.allclose(features, expected)

		dataset = HiddenStateDataset(root, "train", layers=[0], token_positions=[-2])

		assert len(dataset) == 2
		assert dataset.feature_dim == 1  # 1 layer * 1 position * 1 dim

		# Check first item
		features, label = dataset[0]
		assert features.shape == (1,)
		assert label.item() == 1.0
		expected = torch.tensor([2.0])
		assert torch.allclose(features, expected)

		# Check second item
		features, label = dataset[1]
		assert features.shape == (1,)
		assert label.item() == 0.0
		expected = torch.tensor([5.0])
		assert torch.allclose(features, expected)


def test_hidden_state_dataset_negative_indexing_2():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()
		
		file_path = split_dir / "sample_0.pt"
		payload = {
			"features": {
				0: {
					"is_correct": 1.0,
					"hidden_states": {
						0: {0: torch.tensor([1.0]), 2: torch.tensor([3.0])},
					}
				},
				1: {
					"is_correct": 0.0,
					"hidden_states": {
						0: {0: torch.tensor([4.0]), 1: torch.tensor([5.0]), 2: torch.tensor([6.0])},
					}
				},
				2: {
					"is_correct": 1.0,
					"hidden_states": {
						0: {1: torch.tensor([7.0]), 2: torch.tensor([8.0]), 3: torch.tensor([9.0]), 4: torch.tensor([10.0])},
					}
				}
			}
		}
		torch.save(payload, file_path)

		dataset = HiddenStateDataset(root, "train", layers=[0], token_positions=[-3])

		assert len(dataset) == 2

		# Check first item
		features, label = dataset[0]
		assert features.shape == (1,)
		assert label.item() == 0.0
		expected = torch.tensor([4.0])
		assert torch.allclose(features, expected)

		# Check second item
		features, label = dataset[1]
		assert features.shape == (1,)
		assert label.item() == 1.0
		expected = torch.tensor([8.0])
		assert torch.allclose(features, expected)


def test_hidden_state_dataset_average_combiner():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		file_path = split_dir / "sample_0.pt"
		payload = {
			"features": {
				0: {
					"is_correct": 1.0,
					"hidden_states": {
						0: {
							0: torch.tensor([1.0, 1.0]),
							1: torch.tensor([3.0, 3.0]),
						},
					}
				}
			}
		}
		torch.save(payload, file_path)

		dataset = HiddenStateDataset(root, "train", layers=[0], token_positions=[0, 1], combiner="average")

		assert dataset.feature_dim == 2
		features, label = dataset[0]
		assert label.item() == 1.0
		expected = torch.tensor([2.0, 2.0])
		assert torch.allclose(features, expected)


def _make_tail_candidate(label: float, start: int, end: int) -> dict:
	return {
		"is_correct": label,
		"code_token_idx": (start, end),
		"assistant_token_count": end + 2,
		"hidden_states": {
			0: {idx: torch.tensor([float(idx), float(idx) + 0.5]) for idx in range(start, end + 1)},
		},
	}


def test_hidden_state_random_tail_dataset_samples_from_tail():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		file_path = split_dir / "sample_0.pt"
		payload = {
			"features": {
				0: _make_tail_candidate(1.0, 0, 5),
			}
		}
		torch.save(payload, file_path)

		dataset = HiddenStateRandomTailDataset(root, "train", layer=0, tail_fraction=0.5)
		assert len(dataset) == 1

		random.seed(0)
		vector, label = dataset[0]
		assert label.item() == 1.0
		assert vector.shape == (2,)
		# With 6 tokens (0..5) and tail_fraction=0.5, indices 3..5 are eligible.
		assert int(vector[0].item()) in {3, 4, 5}


def test_hidden_state_random_tail_dataset_tail_fraction_one():
	with tempfile.TemporaryDirectory() as tmpdir:
		root = Path(tmpdir)
		split_dir = root / "train"
		split_dir.mkdir()

		file_path = split_dir / "sample_0.pt"
		payload = {
			"features": {
				0: _make_tail_candidate(0.0, 2, 6),
			}
		}
		torch.save(payload, file_path)

		dataset = HiddenStateRandomTailDataset(root, "train", layer=0, tail_fraction=1.0)
		assert len(dataset) == 1

		random.seed(1)
		vector, label = dataset[0]
		assert label.item() == 0.0
		assert vector.shape == (2,)
		# Any position in full sequence 2..6 is allowed.
		assert int(vector[0].item()) in {2, 3, 4, 5, 6}
