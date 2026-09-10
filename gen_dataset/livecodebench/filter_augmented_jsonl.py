"""
Filter augmented LiveCodeBench candidates and optionally derive per-line error maps.

Default behavior:
- Drops candidates flagged by augmentation sentinels/filters:
    `NO_PROGRAM`, `NO_FIX_FOUND`, `line_filter`, and `no_correct_line_filter`.
- Keeps response-correct candidates unless `--filter-response-correct` is set.
- Keeps line-filter behavior enabled unless `--disable-line-filter` is set.
- Removes rows where all candidates are filtered out.
- Writes stats to `<output-path>.stats.json` unless `--stats-path` is provided.

Important options:
- `--add-line-error-map`: add `line_error_map` per remaining candidate.
- `--disable-line-filter`: ignore the `line_filter` flag while preserving other filters.
- `--filter-response-correct`: remove candidates with response-level correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

NO_FIX_FOUND_SENTINEL = "<NO_FIX_FOUND>"
NO_PROGRAM_SENTINEL = "<NO_PROGRAM>"

FILTER_KEYS = (
    "no_program",
    "no_fix_found",
    "line_filter",
    "no_correct_line_filter",
    "response_correct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter augmented LiveCodeBench JSONL candidates and drop empty rows. "
            "Writes filtered JSONL and a stats JSON file."
        )
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Augmented JSONL input path")
    parser.add_argument("--output-path", type=Path, required=True, help="Filtered JSONL output path")
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
        help="Optional stats path; defaults to <output>.stats.json",
    )
    parser.add_argument(
        "--add-line-error-map",
        action="store_true",
        help=(
            "Add per-candidate line_error_map entries after filtering. "
            "Each map has line numbers as keys and booleans indicating whether "
            "that line has no token with label=false (false iff at least one token has label=false)."
        ),
    )
    parser.add_argument(
        "--disable-line-filter",
        action="store_true",
        help="Disable filtering by line_filter while keeping all other filters unchanged.",
    )
    parser.add_argument(
        "--filter-response-correct",
        action="store_true",
        help="Filter out candidates where response-level is_correct is true.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_candidate_line_error_map(candidate_token_labels: object) -> dict[str, bool]:
    line_error_map: dict[str, bool] = {}
    if not isinstance(candidate_token_labels, list):
        return line_error_map

    for token in candidate_token_labels:
        if not isinstance(token, dict):
            continue
        line_no = token.get("line")
        label = token.get("label")
        if line_no is None:
            continue

        line_key = str(line_no)
        line_is_clean = label is not False
        previous = line_error_map.get(line_key, True)
        line_error_map[line_key] = previous and line_is_clean

    return line_error_map


def is_response_correct(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes"}
    return False


def filter_record(
    record: dict,
    stats: dict,
    add_line_error_map: bool = False,
    disable_line_filter: bool = False,
    filter_response_correct: bool = False,
) -> dict | None:
    fixed_programs = record.get("fixed_program", [])
    line_filters = record.get("line_filter", [])
    no_correct_line_filters = record.get("no_correct_line_filter", [])
    response_correct_labels = record.get("is_correct", [])

    if not isinstance(fixed_programs, list):
        fixed_programs = []
    if not isinstance(line_filters, list):
        line_filters = []
    if not isinstance(no_correct_line_filters, list):
        no_correct_line_filters = []
    if not isinstance(response_correct_labels, list):
        response_correct_labels = []

    num_candidates = len(fixed_programs)
    if num_candidates == 0:
        return None

    stats["candidates_total"] += num_candidates

    keep_mask: list[bool] = []
    per_row_filter_hits = {key: False for key in FILTER_KEYS}

    for idx in range(num_candidates):
        fixed_program = fixed_programs[idx]
        line_filter = bool(line_filters[idx]) if idx < len(line_filters) else False
        no_correct_line_filter = bool(no_correct_line_filters[idx]) if idx < len(no_correct_line_filters) else False
        response_correct = (
            is_response_correct(response_correct_labels[idx])
            if idx < len(response_correct_labels)
            else False
        )

        fail_no_program = fixed_program == NO_PROGRAM_SENTINEL
        fail_no_fix = fixed_program == NO_FIX_FOUND_SENTINEL
        fail_line_filter = line_filter and not disable_line_filter
        fail_no_correct_line = no_correct_line_filter
        fail_response_correct = filter_response_correct and response_correct

        should_drop = (
            fail_no_program
            or fail_no_fix
            or fail_line_filter
            or fail_no_correct_line
            or fail_response_correct
        )
        keep_mask.append(not should_drop)

        if fail_no_program:
            stats["candidate_filters"]["no_program"] += 1
            per_row_filter_hits["no_program"] = True
        if fail_no_fix:
            stats["candidate_filters"]["no_fix_found"] += 1
            per_row_filter_hits["no_fix_found"] = True
        if fail_line_filter:
            stats["candidate_filters"]["line_filter"] += 1
            per_row_filter_hits["line_filter"] = True
        if fail_no_correct_line:
            stats["candidate_filters"]["no_correct_line_filter"] += 1
            per_row_filter_hits["no_correct_line_filter"] = True
        if fail_response_correct:
            stats["candidate_filters"]["response_correct"] += 1
            per_row_filter_hits["response_correct"] = True

    if not any(keep_mask):
        stats["candidates_dropped"] += num_candidates
        stats["rows_dropped"] += 1
        for key, hit in per_row_filter_hits.items():
            if hit:
                stats["row_filters"][key] += 1
        return None

    filtered = dict(record)
    for key, value in record.items():
        if isinstance(value, list) and len(value) == num_candidates:
            filtered[key] = [item for item, keep in zip(value, keep_mask) if keep]

    if add_line_error_map:
        token_labels = filtered.get("token_labels", [])
        if isinstance(token_labels, list):
            filtered["line_error_map"] = [
                build_candidate_line_error_map(candidate_tokens)
                for candidate_tokens in token_labels
            ]
        else:
            filtered["line_error_map"] = []

    kept_candidates = sum(keep_mask)
    stats["candidates_kept"] += kept_candidates
    stats["candidates_dropped"] += num_candidates - kept_candidates
    stats["rows_kept"] += 1
    return filtered


def main() -> None:
    args = parse_args()
    stats_path = args.stats_path or args.output_path.with_suffix(args.output_path.suffix + ".stats.json")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "candidate_filters": {key: 0 for key in FILTER_KEYS},
        "row_filters": {key: 0 for key in FILTER_KEYS},
        "rows_total": 0,
        "rows_dropped": 0,
        "rows_kept": 0,
        "candidates_total": 0,
        "candidates_dropped": 0,
        "candidates_kept": 0,
    }

    with args.output_path.open("w", encoding="utf-8") as out_handle:
        for record in iter_jsonl(args.input_path):
            stats["rows_total"] += 1
            filtered = filter_record(
                record,
                stats,
                add_line_error_map=args.add_line_error_map,
                disable_line_filter=args.disable_line_filter,
                filter_response_correct=args.filter_response_correct,
            )
            if filtered is None:
                continue
            out_handle.write(json.dumps(filtered) + "\n")

    with stats_path.open("w", encoding="utf-8") as stats_handle:
        stats_handle.write(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
