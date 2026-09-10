from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import platform
import resource
import signal
import types
import unittest
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate BigCodeBench generations and update is_correct in JSONL",
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to input JSONL with generations")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to overwrite the input file in-place.",
    )
    parser.add_argument("--subset", default="full", help="Dataset subset (default: full)")
    parser.add_argument("--version", default="v0.1.4", help="BigCodeBench dataset version tag")
    parser.add_argument("--eval-timeout", type=int, default=20, help="Timeout per test run (seconds)")
    parser.add_argument(
        "--max-mem-mb",
        type=int,
        default=0,
        help="Per-eval address space/data/stack limit in MB",
    )
    parser.add_argument(
        "--start-method",
        choices=["fork", "spawn", "forkserver"],
        default="fork",
        help="Multiprocessing start method for eval workers",
    )
    parser.add_argument("--start", type=int, default=None, help="Optional task index to start from (inclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of tasks (exclusive)")
    return parser.parse_args()


def _reliability_guard(max_mem_bytes: int) -> None:
    if max_mem_bytes and max_mem_bytes > 0:
        resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (max_mem_bytes, max_mem_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (max_mem_bytes, max_mem_bytes))
    os.environ["OMP_NUM_THREADS"] = "1"
    for attr in [
        "system",
        "popen",
        "fork",
        "forkpty",
        "kill",
        "killpg",
        "remove",
        "removedirs",
        "rmdir",
        "rename",
        "renames",
        "replace",
        "unlink",
        "chroot",
    ]:
        if hasattr(os, attr):
            setattr(os, attr, None)


def _timeout_handler(signum, frame):  # pragma: no cover - signal handler
    raise TimeoutError


def _evaluate_code(test_code: str, code: str, timeout: int, max_mem_bytes: int) -> bool:
    if not code or not code.strip():
        return False
    q: mp.Queue = mp.Queue(maxsize=1)

    def _worker() -> None:
        try:
            _reliability_guard(max_mem_bytes=max_mem_bytes)
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
            module = types.ModuleType("__bigcodebench__")
            exec(code + "\n" + test_code, module.__dict__)
            cases = getattr(module, "TestCases", None)
            if cases is None:
                q.put(False)
                return
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(cases)
            result = unittest.TestResult()
            suite.run(result)
            q.put(not result.errors and not result.failures)
        except MemoryError:
            q.put(False)
        except Exception:
            q.put(False)
        finally:
            signal.alarm(0)

    p = mp.Process(target=_worker)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    try:
        return bool(q.get_nowait())
    except Exception:
        return False


def load_bigcodebench(version: str, subset: str) -> dict:
    extra = f"-{subset}" if subset != "full" else ""
    ds = load_dataset(f"bigcode/bigcodebench{extra}", split=version)
    return {row["task_id"]: row for row in ds} # type: ignore


def stream_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    try:
        mp.set_start_method(args.start_method, force=True)
    except RuntimeError:
        pass

    problems = load_bigcodebench(version=args.version, subset=args.subset)
    out_path = args.output or args.input

    if args.start is not None or args.limit is not None:
        LOGGER.info(
            "Evaluating task indices in range [%s, %s)",
            args.start if args.start is not None else 0,
            args.limit if args.limit is not None else "end",
        )

    updated: List[str] = []
    evaluated = 0
    for idx, rec in enumerate(stream_jsonl(args.input)):
        in_slice = (args.start is None or idx >= args.start) and (args.limit is None or idx < args.limit)
        if not in_slice:
            updated.append(json.dumps(rec))
            continue

        task_id = rec.get("id") or rec.get("task_id")
        problem = problems.get(task_id)
        if not problem:
            LOGGER.warning("Task %s not found in BigCodeBench; skipping", task_id)
            continue
        programs = rec.get("program") or []
        new_labels = [
            _evaluate_code(
                problem.get("test", ""),
                prog or "",
                timeout=args.eval_timeout,
                max_mem_bytes=args.max_mem_mb * 1024 * 1024,
            )
            for prog in programs
        ]
        rec["is_correct"] = new_labels
        LOGGER.info("Task %s: updated is_correct to %s", task_id, new_labels)
        evaluated += 1
        updated.append(json.dumps(rec))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(updated) + ("\n" if updated else ""))
    LOGGER.info("Wrote %d records to %s (evaluated %d)", len(updated), out_path, evaluated)


if __name__ == "__main__":
    main()