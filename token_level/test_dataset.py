import json
from pathlib import Path

import torch

from token_level.dataset import TokenLevelHiddenStateDataset


def _vec(value: float) -> torch.Tensor:
	return torch.tensor([value, value + 0.5], dtype=torch.float32)


def _write_pt(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(payload, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row) + "\n")


def test_token_dataset_searches_all_feature_splits(tmp_path: Path):
	root = tmp_path / "features"
	jsonl_path = tmp_path / "filtered.jsonl"

	payload = {
		"sample_id": "prob-1",
		"features": {
			0: {
				"code_token_idx": [1, 3],
				"hidden_states": {
					48: {1: _vec(1.0), 2: _vec(2.0), 3: _vec(3.0)},
				},
				"token_ids": [10, 11, 12],
			}
		},
	}
	_write_pt(root / "validation" / "prob-1__hs.pt", payload)

	rows = [
		{
			"id": "prob-1",
			"token_ids": [[10, 11, 12]],
			"token_labels": [
				[
					{"label": True, "ws_comment": False},
					{"label": False, "ws_comment": True},
					{"label": True, "ws_comment": False},
				]
			],
		}
	]
	_write_jsonl(jsonl_path, rows)

	dataset = TokenLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=[48],
		drop_ws_comment=True,
	)

	assert len(dataset) == 2
	features0, label0, metadata0 = dataset[0]
	features1, label1, metadata1 = dataset[1]
	assert features0.shape == (2,)
	assert features1.shape == (2,)
	assert label0.item() == 0.0
	assert label1.item() == 0.0
	assert metadata0["line_number"].item() == -1
	assert metadata0["line_label"].item() == -1
	assert metadata1["line_number"].item() == -1
	assert metadata1["line_label"].item() == -1
	assert dataset.stats["tokens_dropped_ws_comment"] == 1
	assert dataset.stats["jsonl_candidates_matched"] == 1


def test_token_dataset_fallbacks_to_fingerprint_when_index_shifted(tmp_path: Path):
	root = tmp_path / "features"
	jsonl_path = tmp_path / "filtered.jsonl"

	payload = {
		"sample_id": "prob-2",
		"features": {
			0: {
				"code_token_idx": [0, 1],
				"hidden_states": {48: {0: _vec(10.0), 1: _vec(11.0)}},
				"token_ids": [1, 1, 1],
			},
			1: {
				"code_token_idx": [0, 1],
				"hidden_states": {48: {0: _vec(20.0), 1: _vec(21.0)}},
				"token_ids": [9, 9, 9],
			},
		},
	}
	_write_pt(root / "train" / "prob-2__hs.pt", payload)

	rows = [
		{
			"id": "prob-2",
			# filtered JSONL compacted candidate list, original target candidate was index 1
			"token_ids": [[9, 9, 9]],
			"token_labels": [[{"label": False, "ws_comment": False}, {"label": True, "ws_comment": False}]],
		}
	]
	_write_jsonl(jsonl_path, rows)

	dataset = TokenLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=[48],
		drop_ws_comment=True,
		match_mode="auto",
	)

	assert len(dataset) == 2
	features0, _, metadata0 = dataset[0]
	features1, _, metadata1 = dataset[1]
	assert torch.allclose(features0, torch.tensor([20.0, 20.5]))
	assert torch.allclose(features1, torch.tensor([21.0, 21.5]))
	assert metadata0["line_number"].item() == -1
	assert metadata0["line_label"].item() == -1
	assert metadata1["line_number"].item() == -1
	assert metadata1["line_label"].item() == -1
	assert dataset.stats["candidate_matches_by_fingerprint"] >= 1


def test_token_dataset_supports_val_alias_for_expected_split(tmp_path: Path):
	root = tmp_path / "features"
	jsonl_path = tmp_path / "filtered.jsonl"

	payload_train = {
		"sample_id": "prob-3",
		"features": {
			0: {
				"code_token_idx": [0, 0],
				"hidden_states": {48: {0: _vec(5.0)}},
				"token_ids": [2],
			}
		},
	}
	payload_val = {
		"sample_id": "prob-3",
		"features": {
			0: {
				"code_token_idx": [0, 0],
				"hidden_states": {48: {0: _vec(8.0)}},
				"token_ids": [2],
			}
		},
	}
	_write_pt(root / "train" / "prob-3__a.pt", payload_train)
	_write_pt(root / "validation" / "prob-3__b.pt", payload_val)

	rows = [{"id": "prob-3", "token_ids": [[2]], "token_labels": [[{"label": True, "ws_comment": False}]]}]
	_write_jsonl(jsonl_path, rows)

	dataset = TokenLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=[48],
		expected_jsonl_split="val",
	)
	features, label, metadata = dataset[0]
	assert torch.allclose(features, torch.tensor([8.0, 8.5]))
	assert label.item() == 0.0
	assert metadata["line_number"].item() == -1
	assert metadata["line_label"].item() == -1
	assert dataset.stats["feature_collisions"] >= 1


def test_token_dataset_returns_line_level_metadata_from_line_error_map(tmp_path: Path):
	root = tmp_path / "features"
	jsonl_path = tmp_path / "filtered.jsonl"

	payload = {
		"sample_id": "prob-4",
		"features": {
			0: {
				"code_token_idx": [0, 2],
				"hidden_states": {
					48: {0: _vec(1.0), 1: _vec(2.0), 2: _vec(3.0)},
				},
				"token_ids": [3, 4, 5],
			}
		},
	}
	_write_pt(root / "train" / "prob-4__hs.pt", payload)

	rows = [
		{
			"id": "prob-4",
			"token_ids": [[3, 4, 5]],
			"token_labels": [[
				{"label": True, "ws_comment": False, "line": 10},
				{"label": False, "ws_comment": False, "line": 10},
				{"label": True, "ws_comment": False, "line": 11},
			]],
			"line_error_map": [
				{
					"10": False,
					"11": True,
				}
			],
		}
	]
	_write_jsonl(jsonl_path, rows)

	dataset = TokenLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=[48],
		drop_ws_comment=True,
	)

	assert len(dataset) == 3
	_, _, metadata0 = dataset[0]
	_, _, metadata1 = dataset[1]
	_, _, metadata2 = dataset[2]

	assert metadata0["line_number"].item() == 10
	assert metadata0["line_label"].item() == 1
	assert metadata1["line_number"].item() == 10
	assert metadata1["line_label"].item() == 1
	assert metadata2["line_number"].item() == 11
	assert metadata2["line_label"].item() == 0
