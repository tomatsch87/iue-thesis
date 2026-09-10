import json
from pathlib import Path

import torch

from .dataset import LineLevelHiddenStateDataset


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


def test_line_dataset_last_token_skips_ws_comment_only_lines(tmp_path: Path):
	root = tmp_path / "features"
	jsonl_path = tmp_path / "filtered.jsonl"

	payload = {
		"sample_id": "prob-line-1",
		"features": {
			0: {
				"code_token_idx": [0, 4],
				"hidden_states": {
					48: {
						0: _vec(1.0),
						1: _vec(2.0),
						2: _vec(3.0),
						3: _vec(4.0),
						4: _vec(5.0),
					},
				},
				"token_ids": [10, 11, 12, 13, 14],
			}
		},
	}
	_write_pt(root / "train" / "prob-line-1__hs.pt", payload)

	rows = [
		{
			"id": "prob-line-1",
			"token_ids": [[10, 11, 12, 13, 14]],
			"token_labels": [[
				{"label": True, "ws_comment": False, "line": 10},
				{"label": True, "ws_comment": False, "line": 10},
				{"label": True, "ws_comment": True, "line": 11},
				{"label": True, "ws_comment": False, "line": 12},
				{"label": True, "ws_comment": False, "line": 12},
			]],
			"line_error_map": [
				{
					"10": False,
					"11": True,
					"12": True,
				}
			],
		}
	]
	_write_jsonl(jsonl_path, rows)

	dataset = LineLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=[48],
		line_representation="last_token",
		drop_ws_comment=True,
	)

	assert len(dataset) == 2
	features0, label0, metadata0 = dataset[0]
	features1, label1, metadata1 = dataset[1]

	assert torch.allclose(features0, torch.tensor([2.0, 2.5]))
	assert torch.allclose(features1, torch.tensor([5.0, 5.5]))
	assert label0.item() == 1.0
	assert label1.item() == 0.0
	assert metadata0["line_number"].item() == 10
	assert metadata1["line_number"].item() == 12
	assert metadata0["token_count"].item() == 2
	assert metadata1["token_count"].item() == 2
	assert dataset.stats["tokens_dropped_ws_comment"] == 1


def test_line_dataset_aggregate_mean_uses_non_ws_comment_tokens(tmp_path: Path):
	root = tmp_path / "features"
	jsonl_path = tmp_path / "filtered.jsonl"

	payload = {
		"sample_id": "prob-line-2",
		"features": {
			0: {
				"code_token_idx": [0, 2],
				"hidden_states": {
					48: {
						0: _vec(2.0),
						1: _vec(4.0),
						2: _vec(9.0),
					},
				},
				"token_ids": [20, 21, 22],
			}
		},
	}
	_write_pt(root / "validation" / "prob-line-2__hs.pt", payload)

	rows = [
		{
			"id": "prob-line-2",
			"token_ids": [[20, 21, 22]],
			"token_labels": [[
				{"label": True, "ws_comment": False, "line": 30},
				{"label": True, "ws_comment": False, "line": 30},
				{"label": True, "ws_comment": True, "line": 31},
			]],
			"line_error_map": [{"30": False}],
		}
	]
	_write_jsonl(jsonl_path, rows)

	dataset = LineLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=[48],
		line_representation="aggregate",
		aggregation="mean",
		drop_ws_comment=True,
	)

	assert len(dataset) == 1
	features, label, metadata = dataset[0]
	assert torch.allclose(features, torch.tensor([3.0, 3.5]))
	assert label.item() == 1.0
	assert metadata["line_number"].item() == 30
	assert metadata["token_count"].item() == 2


def test_line_dataset_derives_line_label_when_line_error_map_missing(tmp_path: Path):
	root = tmp_path / "features"
	jsonl_path = tmp_path / "filtered.jsonl"

	payload = {
		"sample_id": "prob-line-3",
		"features": {
			0: {
				"code_token_idx": [0, 1],
				"hidden_states": {48: {0: _vec(7.0), 1: _vec(8.0)}},
				"token_ids": [31, 32],
			}
		},
	}
	_write_pt(root / "train" / "prob-line-3__hs.pt", payload)

	rows = [
		{
			"id": "prob-line-3",
			"token_ids": [[31, 32]],
			"token_labels": [[
				{"label": True, "ws_comment": False, "line": 50},
				{"label": False, "ws_comment": False, "line": 50},
			]],
		}
	]
	_write_jsonl(jsonl_path, rows)

	dataset = LineLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=[48],
		line_representation="last_token",
		drop_ws_comment=True,
	)

	assert len(dataset) == 1
	_, label, metadata = dataset[0]
	assert label.item() == 1.0
	assert metadata["line_number"].item() == 50


def test_line_dataset_fallbacks_to_fingerprint_when_index_shifted(tmp_path: Path):
	root = tmp_path / "features"
	jsonl_path = tmp_path / "filtered.jsonl"

	payload = {
		"sample_id": "prob-line-4",
		"features": {
			0: {
				"code_token_idx": [0, 0],
				"hidden_states": {48: {0: _vec(1.0)}},
				"token_ids": [1, 1, 1],
			},
			1: {
				"code_token_idx": [0, 0],
				"hidden_states": {48: {0: _vec(9.0)}},
				"token_ids": [9, 9, 9],
			},
		},
	}
	_write_pt(root / "train" / "prob-line-4__hs.pt", payload)

	rows = [
		{
			"id": "prob-line-4",
			"token_ids": [[9, 9, 9]],
			"token_labels": [[{"label": False, "ws_comment": False, "line": 77}]],
			"line_error_map": [{"77": True}],
		}
	]
	_write_jsonl(jsonl_path, rows)

	dataset = LineLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=[48],
		line_representation="last_token",
		drop_ws_comment=True,
		match_mode="auto",
	)

	assert len(dataset) == 1
	features, label, metadata = dataset[0]
	assert torch.allclose(features, torch.tensor([9.0, 9.5]))
	assert label.item() == 0.0
	assert metadata["line_number"].item() == 77
	assert dataset.stats["candidate_matches_by_fingerprint"] >= 1


def test_line_dataset_line_error_map_true_clean_false_error(tmp_path: Path):
	root = tmp_path / "features"
	jsonl_path = tmp_path / "filtered.jsonl"

	payload = {
		"sample_id": "prob-line-5",
		"features": {
			0: {
				"code_token_idx": [0, 1],
				"hidden_states": {48: {0: _vec(3.0), 1: _vec(6.0)}},
				"token_ids": [41, 42],
			}
		},
	}
	_write_pt(root / "train" / "prob-line-5__hs.pt", payload)

	rows = [
		{
			"id": "prob-line-5",
			"token_ids": [[41, 42]],
			"token_labels": [[
				{"label": True, "ws_comment": False, "line": 100},
				{"label": True, "ws_comment": False, "line": 101},
			]],
			"line_error_map": [{"100": False, "101": True}],
		}
	]
	_write_jsonl(jsonl_path, rows)

	dataset = LineLevelHiddenStateDataset(
		root=root,
		jsonl_path=jsonl_path,
		layers=[48],
		line_representation="last_token",
		drop_ws_comment=True,
	)

	assert len(dataset) == 2
	_, label0, metadata0 = dataset[0]
	_, label1, metadata1 = dataset[1]

	assert metadata0["line_number"].item() == 100
	assert label0.item() == 1.0
	assert metadata1["line_number"].item() == 101
	assert label1.item() == 0.0
