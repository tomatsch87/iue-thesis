"""
Flatten BigCodeBench generation JSONL into sanitizer-ready `task_id/solution` rows.

Default behavior:
- Reads `program` list from each input record and writes one output line per program.
- Uses `id` as task identifier, with fallback to `task_id`.
- Skips records without a valid task id or program sequence.

Important options:
- `--program-field`: select a different candidate list field (default `program`).
- `--id-field`: select a different id field (default `id`).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten BigCodeBench generation JSONL into sanitizer-ready samples (task_id + solution per line)",
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to the source JSONL (build_dataset output)")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL where flattened samples will be written",
    )
    parser.add_argument(
        "--program-field",
        default="program",
        help="Field name containing the list of code candidates (default: program)",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="Field name containing the task id (falls back to task_id if missing)",
    )
    return parser.parse_args()


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    samples = []
    for idx, rec in enumerate(stream_jsonl(args.input)):
        task_id = rec.get(args.id_field) or rec.get("task_id")
        programs = rec.get(args.program_field) or []

        if not task_id:
            LOGGER.warning("Record %s missing task_id; skipping", idx)
            continue
        if not isinstance(programs, Sequence):
            LOGGER.warning("Record %s %s not a sequence; skipping", idx, args.program_field)
            continue

        for prog in programs:
            samples.append({"task_id": task_id, "solution": prog or ""})

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    LOGGER.info("Wrote %d samples to %s", len(samples), out_path)


if __name__ == "__main__":
    main()