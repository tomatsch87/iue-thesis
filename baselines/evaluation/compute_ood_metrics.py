from __future__ import annotations

import argparse
import math
import os
import random
import re
from statistics import mean, stdev
from typing import Dict, List, Tuple


FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_markdown_tables(text: str) -> Dict[str, List[Dict[str, str]]]:
    lines = text.splitlines()
    i = 0
    tables: Dict[str, List[Dict[str, str]]] = {}
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            model = line.replace("### ", "").strip()
            i += 1
            # Find header line
            while i < len(lines) and not lines[i].startswith("|"):
                i += 1
            if i >= len(lines):
                break
            header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            i += 2  # skip header + separator
            rows: List[Dict[str, str]] = []
            while i < len(lines) and lines[i].startswith("|"):
                cols = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(cols) == len(header):
                    rows.append(dict(zip(header, cols)))
                i += 1
            tables[model] = rows
        else:
            i += 1
    return tables


def extract_float(value: str) -> float:
    value = value.replace("**", "").replace("*", "").strip()
    match = FLOAT_RE.search(value)
    if not match:
        raise ValueError(f"No float found in value: {value}")
    return float(match.group(0))


def best_auc(rows: List[Dict[str, str]], exclude_bcb_rank: bool) -> Tuple[float, str]:
    best = None
    best_name = ""
    for row in rows:
        rank = row.get("rank", "").strip()
        if exclude_bcb_rank and rank == "BCB":
            continue
        auc = extract_float(row.get("roc_auc", ""))
        name = row.get("Baseline Model", "")
        if best is None or auc > best:
            best = auc
            best_name = name
    if best is None:
        raise ValueError("No AUROC rows found after filtering")
    return best, best_name


def bootstrap_ci(values: List[float], samples: int, seed: int) -> Tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(resample))
    means.sort()
    lo_idx = max(0, int(0.025 * samples) - 1)
    hi_idx = min(samples - 1, int(0.975 * samples) - 1)
    return means[lo_idx], means[hi_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default=os.path.dirname(__file__), help="Directory with eval markdown files")
    parser.add_argument("--id-file", default="bcb_instruct_eval.md", help="ID (full dataset) markdown file")
    parser.add_argument("--domain-glob", default="bcb_instruct_domain_*.md", help="Glob for domain markdown files")
    parser.add_argument("--epsilon", type=float, default=0.02, help="Threshold for abs drop fraction")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_dir = args.eval_dir
    id_path = os.path.join(eval_dir, args.id_file)
    if not os.path.exists(id_path):
        raise FileNotFoundError(id_path)

    with open(id_path, "r", encoding="utf-8") as f:
        id_tables = parse_markdown_tables(f.read())

    id_best: Dict[str, Tuple[float, str]] = {}
    for model, rows in id_tables.items():
        id_best[model] = best_auc(rows, exclude_bcb_rank=False)

    domain_files = [
        os.path.join(eval_dir, name)
        for name in os.listdir(eval_dir)
        if name.startswith("bcb_instruct_domain_") and name.endswith(".md")
    ]
    domain_files.sort()

    per_model_shortfalls: Dict[str, List[float]] = {m: [] for m in id_best.keys()}
    per_model_signed_drops: Dict[str, List[float]] = {m: [] for m in id_best.keys()}

    print("# OOD generalization metrics\n")
    print(f"Epsilon: {args.epsilon}")
    print(f"Domains: {len(domain_files)}")
    print("")

    for domain_path in domain_files:
        domain_name = os.path.splitext(os.path.basename(domain_path))[0].replace("bcb_domain_", "")
        with open(domain_path, "r", encoding="utf-8") as f:
            domain_tables = parse_markdown_tables(f.read())

        print(f"## Domain: {domain_name}")
        print("| Model | ID best AUROC | OOD best AUROC | Signed drop (OOD-ID) | Shortfall (max(0, ID-OOD)) |")
        print("|---|---:|---:|---:|---:|")
        for model, (id_auc, _) in id_best.items():
            if model not in domain_tables:
                continue
            ood_auc, _ = best_auc(domain_tables[model], exclude_bcb_rank=True)
            signed_drop = ood_auc - id_auc
            shortfall = max(0.0, id_auc - ood_auc)
            per_model_shortfalls[model].append(shortfall)
            per_model_signed_drops[model].append(signed_drop)
            print(f"| {model} | {id_auc:.6f} | {ood_auc:.6f} | {signed_drop:.6f} | {shortfall:.6f} |")
        print("")

    print("## Summary (per model)")
    print("| Model | Mean shortfall | Std shortfall | 95% CI mean shortfall | Fraction shortfall < epsilon | Mean signed drop | Std signed drop |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model, shortfalls in per_model_shortfalls.items():
        if not shortfalls:
            continue
        signed_drops = per_model_signed_drops[model]
        mean_shortfall = mean(shortfalls)
        std_shortfall = stdev(shortfalls) if len(shortfalls) > 1 else 0.0
        ci_lo, ci_hi = bootstrap_ci(shortfalls, args.bootstrap_samples, args.seed)
        frac = sum(1 for d in shortfalls if d < args.epsilon) / len(shortfalls)
        signed_mean = mean(signed_drops)
        signed_std = stdev(signed_drops) if len(signed_drops) > 1 else 0.0
        print(
            f"| {model} | {mean_shortfall:.6f} | {std_shortfall:.6f} | [{ci_lo:.6f}, {ci_hi:.6f}] | {frac:.3f} | {signed_mean:.6f} | {signed_std:.6f} |"
        )

    # Overall pooled summary
    all_shortfalls = [d for v in per_model_shortfalls.values() for d in v]
    all_signed = [d for v in per_model_signed_drops.values() for d in v]
    if all_shortfalls:
        mean_shortfall = mean(all_shortfalls)
        std_shortfall = stdev(all_shortfalls) if len(all_shortfalls) > 1 else 0.0
        ci_lo, ci_hi = bootstrap_ci(all_shortfalls, args.bootstrap_samples, args.seed)
        frac = sum(1 for d in all_shortfalls if d < args.epsilon) / len(all_shortfalls)
        signed_mean = mean(all_signed)
        signed_std = stdev(all_signed) if len(all_signed) > 1 else 0.0
        print("\n## Summary (all models pooled)")
        print("| Mean shortfall | Std shortfall | 95% CI mean shortfall | Fraction shortfall < epsilon | Mean signed drop | Std signed drop |")
        print("|---:|---:|---:|---:|---:|---:|")
        print(
            f"| {mean_shortfall:.6f} | {std_shortfall:.6f} | [{ci_lo:.6f}, {ci_hi:.6f}] | {frac:.3f} | {signed_mean:.6f} | {signed_std:.6f} |"
        )


if __name__ == "__main__":
    main()