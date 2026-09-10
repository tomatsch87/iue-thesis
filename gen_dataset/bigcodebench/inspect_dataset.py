from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect BigCodeBench JSONL dataset")
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Path to bigcodebench_<model>.jsonl",
    )
    parser.add_argument(
        "--save-prefix",
        default="dataset",
        help="Prefix for saved plot filenames",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Where to save plots (defaults to JSONL directory)",
    )
    parser.add_argument(
        "--bcb-dataset",
        default="bigcode/bigcodebench",
        help="BigCodeBench dataset name",
    )
    parser.add_argument(
        "--bcb-config",
        default="default",
        help="BigCodeBench dataset config suffix (optional)",
    )
    parser.add_argument(
        "--bcb-version",
        default="v0.1.4",
        help="BigCodeBench dataset version used as split",
    )
    parser.add_argument(
        "--domain-map",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "feat_extract" / "bcb_domains.json",
        help="Path to bcb_domains.json mapping libs to domains",
    )
    return parser.parse_args()


def bucket_correctness(program: str, is_correct: bool) -> str:
    if is_correct:
        return "correct"
    return "no_program" if not program.strip() else "tests_failed"


def plot_label_balance(*, bucket_counts: Dict[str, int], save_path: Path) -> None:
    labels = ["correct", "tests_failed", "no_program"]
    counts = [bucket_counts.get(label, 0) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, counts, color=["tab:green", "tab:red", "tab:gray"])
    ax.set_title("Label balance")
    ax.set_ylabel("Outputs")

    for idx, count in enumerate(counts):
        ax.text(idx, count, str(count), ha="center", va="bottom")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)


def normalize_libs(raw: object) -> List[str]:
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw if isinstance(item, str)]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed if isinstance(item, str)]
        except (SyntaxError, ValueError):
            pass
        if "," in text:
            parts = [p.strip().strip("'\"") for p in text.strip("[]").split(",")]
            return [p for p in parts if p]
        return [text.strip("'\"")]
    return []


def load_domain_map(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}; got {type(data)}")
    return {str(k): str(v) for k, v in data.items()}


def build_task_domain_index(dataset_name: str, dataset_config: str, version: str, lib_to_domain: Dict[str, str]) -> Dict[str, List[str]]:
    if dataset_config and dataset_config != "default":
        dataset_name = f"{dataset_name}-{dataset_config}"
    dataset = load_dataset(dataset_name, split=version)
    lib_to_domain_lower = {k.lower(): v for k, v in lib_to_domain.items()}
    index: Dict[str, List[str]] = {}

    for sample in dataset:
        if not isinstance(sample, dict):
            continue
        task_id = sample.get("task_id") or sample.get("id")
        if task_id is None:
            continue
        libs = normalize_libs(sample.get("libs"))
        domains: List[str] = []
        for lib in libs:
            domain = lib_to_domain.get(lib) or lib_to_domain_lower.get(lib.lower())
            if domain and domain not in domains:
                domains.append(domain)
        index[str(task_id)] = domains
    return index


def plot_domain_balance(*, domain_counts: Dict[str, Counter[str]], save_path: Path) -> None:
    labels = ["correct", "tests_failed", "no_program"]
    domains = sorted(domain_counts.keys())
    if not domains:
        return
    counts = {label: [domain_counts[d].get(label, 0) for d in domains] for label in labels}

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(domains)), 6))
    x = range(len(domains))
    width = 0.25
    ax.bar([i - width for i in x], counts["correct"], width, label="correct", color="tab:green")
    ax.bar(x, counts["tests_failed"], width, label="tests_failed", color="tab:red")
    ax.bar([i + width for i in x], counts["no_program"], width, label="no_program", color="tab:gray")

    ax.set_xticks(list(x))
    ax.set_xticklabels(domains, rotation=45, ha="right")
    ax.set_title("Label balance by domain")
    ax.set_ylabel("Outputs")
    ax.legend()

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    path: Path = args.path
    if not path.exists():
        raise FileNotFoundError(path)
    output_dir = args.output_dir or path.parent

    total_records = 0
    empty_program_entries = 0
    ids_with_empty: List[str] = []
    bucket_counts: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            rec = json.loads(line)
            programs: List[str] = list(rec.get("program") or [])
            outputs: List[str] = list(rec.get("output") or [])
            flags: List[bool] = list(rec.get("is_correct") or [])

            empty_indices = [i for i, p in enumerate(programs) if not p or not str(p).strip()]
            if empty_indices:
                ids_with_empty.append(str(rec.get("id")))
                empty_program_entries += len(empty_indices)
                for i in empty_indices:
                    print(f"Record ID {rec.get('id')} has empty programs at index: {i}")
                    output = outputs[i] if i < len(outputs) else "N/A"
                    print(f"Output (last 100 chars): ...{output[-100:]}")

            num_candidates = max(len(programs), len(outputs), len(flags))
            if num_candidates == 0:
                continue
            for idx in range(num_candidates):
                program_text = programs[idx] if idx < len(programs) else ""
                flag = bool(flags[idx]) if idx < len(flags) else False
                bucket = bucket_correctness(program_text, flag)
                bucket_counts[bucket] += 1

    domain_counts: Dict[str, Counter[str]] = {}
    domain_map = load_domain_map(args.domain_map)
    task_domains = build_task_domain_index(args.bcb_dataset, args.bcb_config, args.bcb_version, domain_map)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task_id = str(rec.get("id"))
            domains = task_domains.get(task_id, [])
            if not domains:
                continue
            programs = list(rec.get("program") or [])
            flags = list(rec.get("is_correct") or [])
            num_candidates = max(len(programs), len(flags))
            if num_candidates == 0:
                continue
            for idx in range(num_candidates):
                program_text = programs[idx] if idx < len(programs) else ""
                flag = bool(flags[idx]) if idx < len(flags) else False
                bucket = bucket_correctness(program_text, flag)
                for domain in domains:
                    domain_counts.setdefault(domain, Counter())[bucket] += 1

    print({
        "total_records": total_records,
        "records_with_empty_program": len(ids_with_empty),
        "empty_program_entries": empty_program_entries,
        "label_counts": dict(bucket_counts),
    })
    print("ids_with_empty_programs:", ids_with_empty)

    plot_path = output_dir / f"{args.save_prefix}_label_balance.png"
    plot_label_balance(bucket_counts=bucket_counts, save_path=plot_path)
    print(f"Saved label balance plot to {plot_path}")

    domain_plot_path = output_dir / f"{args.save_prefix}_domain_label_balance.png"
    plot_domain_balance(domain_counts=domain_counts, save_path=domain_plot_path)
    print(f"Saved domain label balance plot to {domain_plot_path}")


if __name__ == "__main__":
    main()