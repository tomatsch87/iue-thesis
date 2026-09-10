"""
Re-evaluate BigCodeBench generations with the official Docker evaluator.

Default behavior:
- Reads input JSONL and overwrites it unless `--output` is provided.
- Uses split `instruct`, subset `full`, and image
    `bigcodebench/bigcodebench-evaluate:latest`.
- Runs Docker evaluation with `parallel=2`, `min_time_limit=1`, `pass_k=1`,
    and calibration disabled (`--calibrated` false).
- Supports both build-dataset format (task + program list) and sanitizer format
    (one `task_id`/`solution` per row).
- Updates `is_correct` from Docker status results and writes merged JSONL.

Important options:
- Docker controls: `--docker-binary`, `--docker-image`, `--docker-extra-arg`.
- Limits/performance: `--parallel`, `--min-time-limit`, `--max-as-limit`,
    `--max-data-limit`, `--max-stack-limit`.
- Dataset scope: `--split`, `--subset`, `--start`, `--limit`.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate BigCodeBench generations via the official Docker sandbox "
            "and update is_correct in JSONL"
        ),
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to input JSONL with generations")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to overwrite the input file in-place.",
    )
    parser.add_argument("--split", choices=["instruct", "complete"], default="instruct", help="BigCodeBench split")
    parser.add_argument("--subset", choices=["full", "hard"], default="full", help="BigCodeBench subset")
    parser.add_argument(
        "--docker-image",
        default="bigcodebench/bigcodebench-evaluate:latest",
        help="Docker image used for evaluation",
    )
    parser.add_argument("--docker-binary", default="docker", help="Docker CLI binary")
    parser.add_argument(
        "--docker-extra-arg",
        action="append",
        default=[],
        help="Additional args passed to docker run (repeatable)",
    )
    parser.add_argument("--parallel", type=int, default=2, help="Number of parallel workers inside the container")
    parser.add_argument("--min-time-limit", type=int, default=1, help="Minimum timeout per task (seconds)")
    parser.add_argument("--max-as-limit", type=int, default=30 * 1024, help="Address space limit (MB)")
    parser.add_argument("--max-data-limit", type=int, default=30 * 1024, help="Data segment limit (MB)")
    parser.add_argument("--max-stack-limit", type=int, default=10, help="Stack limit (MB)")
    parser.add_argument("--pass-k", default="1", help="Pass@k values, comma-separated")
    parser.add_argument(
        "--calibrated",
        action="store_true",
        default=False,
        help="Keep BigCodeBench calibration prefix (default: False to mirror raw code execution)",
    )
    parser.add_argument("--start", type=int, default=None, help="Optional task index to start from (inclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of tasks (exclusive)")
    return parser.parse_args()


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _status_to_bool(status: str | None) -> bool:
    if status is None:
        return False
    return str(status).lower() in {"pass", "passed", "ok", "success"}


def _build_samples(input_path: Path, start: int | None, limit: int | None):
    """Collect samples to evaluate and track how to map results back.

    Supports two input shapes:
    1) build_dataset-style: one record per task with `program` list of candidates.
    2) sanitizer-style: one record per candidate with `solution` and `task_id`.
    """

    updated_records: List[dict] = []
    pending = []
    samples: List[dict] = []
    task_ids = []

    for idx, rec in enumerate(stream_jsonl(input_path)):
        in_slice = (start is None or idx >= start) and (limit is None or idx < limit)
        if not in_slice:
            continue

        task_id = rec.get("id") or rec.get("task_id")
        programs = rec.get("program")
        solution = rec.get("solution")

        if not task_id:
            LOGGER.warning("Record %s missing task_id; leaving untouched", idx)
            updated_records.append(rec)
            continue

        # Sanitized format: one solution per record
        if programs is None and solution is not None:
            samples.append({"task_id": task_id, "solution": solution or ""})
            pending.append((task_id, 1, rec))
            task_ids.append(task_id)
            continue

        # Legacy format: list of programs per task
        if not isinstance(programs, Sequence):
            LOGGER.warning("Record %s programs not a sequence; leaving untouched", idx)
            updated_records.append(rec)
            continue

        for prog in programs:
            samples.append({"task_id": task_id, "solution": prog or ""})

        pending.append((task_id, len(programs), rec))
        task_ids.append(task_id)

    return updated_records, pending, samples, task_ids


def _write_samples(samples: List[dict], workdir: Path) -> Path:
    samples_path = workdir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    return samples_path


def _run_docker(args: argparse.Namespace, samples_path: Path, task_ids: List[str]) -> Path:
    if not task_ids:
        raise ValueError("No tasks selected for evaluation")

    container_samples = f"/app/{samples_path.name}"
    result_path = samples_path.with_name(samples_path.name.replace(".jsonl", "_eval_results.json"))
    selective = ",".join(sorted(set(task_ids)))

    pass_k = args.pass_k
    if isinstance(pass_k, (list, tuple)):
        pass_k = ",".join(str(k) for k in pass_k)
    else:
        pass_k = str(pass_k)
    if "," not in pass_k:
        pass_k = f"{pass_k},"

    cmd = [
        args.docker_binary,
        "run",
        "--rm",
        "-v",
        f"{samples_path.parent}:/app",
    ]
    cmd.extend(args.docker_extra_arg)
    cmd.append(args.docker_image)
    cmd.extend(
        [
            "--execution",
            "local",
            "--split",
            args.split,
            "--subset",
            args.subset,
            "--samples",
            container_samples,
            "--selective_evaluate",
            selective,
            "--parallel",
            str(args.parallel),
            "--pass_k",
            pass_k,
            "--save_pass_rate",
            "False",
            "--min_time_limit",
            str(args.min_time_limit),
            "--max_as_limit",
            str(args.max_as_limit),
            "--max_data_limit",
            str(args.max_data_limit),
            "--max_stack_limit",
            str(args.max_stack_limit),
            "--calibrated",
            str(bool(args.calibrated)),
        ]
    )

    LOGGER.info("Running Docker evaluation: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    return result_path


def _load_results(result_path: Path) -> Dict[str, List[dict]]:
    with result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    eval_section = data.get("eval", {})
    if not isinstance(eval_section, dict):
        raise ValueError("Unexpected eval result format")
    return eval_section


def _update_records(updated: List[dict], pending, eval_results: Dict[str, List[dict]]):
    cursor = {tid: 0 for tid in eval_results.keys()}
    for task_id, count, rec in pending:
        results = eval_results.get(task_id, [])
        offset = cursor.get(task_id, 0)
        slice_results = results[offset : offset + count]
        cursor[task_id] = offset + count

        statuses = [_status_to_bool(r.get("status")) for r in slice_results]
        if len(statuses) < count:
            missing = count - len(statuses)
            LOGGER.warning("Task %s returned %s/%s results; padding failures", task_id, len(statuses), count)
            statuses.extend([False] * missing)

        if count == 1:
            rec["is_correct"] = statuses[0]
        else:
            rec["is_correct"] = statuses
        updated.append(rec)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    updated_records, pending, samples, task_ids = _build_samples(args.input, args.start, args.limit)

    if pending:
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            samples_path = _write_samples(samples, workdir)
            result_path = _run_docker(args, samples_path, task_ids)
            eval_results = _load_results(result_path)
            _update_records(updated_records, pending, eval_results)
    else:
        LOGGER.info("No records selected for evaluation; writing output unchanged")

    out_path = args.output or args.input
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in updated_records) + ("\n" if updated_records else ""))
    LOGGER.info("Wrote %d records to %s", len(updated_records), out_path)


if __name__ == "__main__":
    main()
    # Sanitize (expects "task_id" and "solution" fields)
    # docker run --rm -v "$PWD":/app \
    # --entrypoint bigcodebench.sanitize \
    # bigcodebench/bigcodebench-evaluate:latest \
    # --samples /app/gen_dataset/bigcodebench/raw_samples.jsonl

    # Evaluate
    # python eval_jsonl_docker.py \
    # --input samples_sanitized.jsonl \
    # --output samples_sanitized_scored.jsonl \
    # --docker-extra-arg="--memory=16g"