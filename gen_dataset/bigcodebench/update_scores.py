"""
Merge scored BigCodeBench `is_correct` results back into generation JSONL.

Default behavior:
- Reads original generation JSONL (`--input`) and scored per-solution JSONL (`--scores`).
- Matches scores by task id in stream order and updates each record's `is_correct`.
- Overwrites input by default unless `--output` is provided.
- In non-strict mode (default), score/program count mismatches are warned and aligned
    by truncating or padding with `False`.

Important options:
- `--strict`: raise an error on any score-count mismatch.
- Supports task id keys from either `task_id` or `id` in score records.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply scored is_correct labels back onto generation JSONL")
    parser.add_argument("--input", required=True, type=Path, help="Path to original generation JSONL (from build_dataset)")
    parser.add_argument("--scores", required=True, type=Path, help="Path to scored JSONL with per-solution is_correct")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: overwrite input)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Error if score counts do not match program counts (default: warn and align)",
    )
    return parser.parse_args()


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).lower() in {"true", "1", "yes", "pass", "passed", "ok", "success"}


def load_scores(scores_path: Path) -> Dict[str, List[bool]]:
    grouped: Dict[str, List[bool]] = {}
    for rec in stream_jsonl(scores_path):
        task_id = rec.get("task_id") or rec.get("id")
        if task_id is None:
            LOGGER.warning("Score record missing task_id/id; skipping: %s", rec)
            continue
        grouped.setdefault(task_id, []).append(_to_bool(rec.get("is_correct")))
    return grouped


def take_slice(task_id: str, scores: Dict[str, List[bool]], cursor: Dict[str, int], expected: int) -> List[bool]:
    items = scores.get(task_id, [])
    start = cursor.get(task_id, 0)
    end = start + expected
    slice_vals = items[start:end]
    cursor[task_id] = start + expected
    return slice_vals


def align_lengths(task_id: str, slice_vals: List[bool], expected: int, strict: bool) -> List[bool]:
    if len(slice_vals) == expected:
        return slice_vals
    msg = "Task %s score count %d != program count %d" % (task_id, len(slice_vals), expected)
    if strict:
        raise ValueError(msg)
    LOGGER.warning("%s; aligning by truncating/padding", msg)
    if len(slice_vals) > expected:
        return slice_vals[:expected]
    # pad with False if too short
    return slice_vals + [False] * (expected - len(slice_vals))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    scores = load_scores(args.scores)
    cursor: Dict[str, int] = {}

    updated: List[str] = []
    for rec in stream_jsonl(args.input):
        task_id = rec.get("id") or rec.get("task_id")
        if task_id is None:
            LOGGER.warning("Input record missing id/task_id; leaving unchanged")
            updated.append(json.dumps(rec))
            continue

        programs = rec.get("program")
        if isinstance(programs, list):
            expected = len(programs)
        else:
            expected = 1

        slice_vals = take_slice(task_id, scores, cursor, expected)
        slice_vals = align_lengths(task_id, slice_vals, expected, args.strict)

        rec["is_correct"] = slice_vals if expected != 1 else slice_vals[0]
        updated.append(json.dumps(rec))

    out_path = args.output or args.input
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(updated) + ("\n" if updated else ""))

    # Warn about any unused scores
    for tid, vals in scores.items():
        used = cursor.get(tid, 0)
        if used < len(vals):
            LOGGER.warning("Unused scores for task %s: %d/%d consumed", tid, used, len(vals))

    LOGGER.info("Wrote %d records to %s", len(updated), out_path)


if __name__ == "__main__":
    main()