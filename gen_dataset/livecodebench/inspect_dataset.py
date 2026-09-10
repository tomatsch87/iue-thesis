from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Inspect LiveCodeBench JSONL dataset")
	parser.add_argument(
		"--path",
		type=Path,
		required=True,
		help="Path to livecodebench_<model>.jsonl",
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
	return parser.parse_args()


def bucket_correctness(program: str, is_correct: bool) -> str:
	if is_correct:
		return "correct"
	return "no_program" if not program.strip() else "tests_failed"


def plot_distributions(
	*,
	difficulty_counts: Counter[str],
	per_difficulty_correctness: Dict[str, Dict[str, int]],
	save_path: Path,
) -> None:
	difficulties = sorted(difficulty_counts.keys())
	if not difficulties:
		return
	problem_counts = [difficulty_counts[d] for d in difficulties]
	correct_counts = [per_difficulty_correctness.get(d, {}).get("correct", 0) for d in difficulties]
	no_program_counts = [per_difficulty_correctness.get(d, {}).get("no_program", 0) for d in difficulties]
	tests_failed_counts = [per_difficulty_correctness.get(d, {}).get("tests_failed", 0) for d in difficulties]

	fig, ax_arr = plt.subplots(1, 2, figsize=(14, 5))
	fig.suptitle("Dataset distributions")

	ax_left = ax_arr[0]
	ax_left.bar(difficulties, problem_counts, color="tab:blue")
	ax_left.set_title("Problems per difficulty")
	ax_left.set_ylabel("Problems")
	ax_left.set_xlabel("Difficulty")

	ax_right = ax_arr[1]
	ax_right.bar(difficulties, no_program_counts, label="No program", color="tab:gray")
	ax_right.bar(
		difficulties,
		tests_failed_counts,
		bottom=no_program_counts,
		label="Tests failed",
		color="tab:red",
	)
	stacked_bottom = [n + t for n, t in zip(no_program_counts, tests_failed_counts)]
	ax_right.bar(
		difficulties,
		correct_counts,
		bottom=stacked_bottom,
		label="Correct outputs",
		color="tab:green",
	)
	ax_right.set_title("Output outcomes by difficulty")
	ax_right.set_ylabel("Outputs")
	ax_right.legend()

	fig.tight_layout(rect=(0, 0, 1, 0.95))
	save_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(save_path))
	plt.close(fig)


def main() -> None:
	args = parse_args()
	path: Path = args.path
	if not path.exists():
		raise FileNotFoundError(path)
	output_dir = args.output_dir or path.parent

	ids_with_empty: List[str] = []
	total_records = 0
	empty_program_entries = 0
	difficulty_counts: Counter[str] = Counter()
	per_difficulty_correctness: Dict[str, Dict[str, int]] = defaultdict(
		lambda: {"correct": 0, "no_program": 0, "tests_failed": 0}
	)

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
			difficulty = str(rec.get("difficulty", "unknown")).lower()
			difficulty_counts[difficulty] += 1

			empty_indices = [i for i, p in enumerate(programs) if not p or not str(p).strip()]
			if empty_indices:
				ids_with_empty.append(str(rec.get("id")))
				empty_program_entries += len(empty_indices)
				for i in empty_indices:
					print(f"Record ID {rec.get('id')} has empty programs at index: {i}")
					output = outputs[i] if i < len(outputs) else "N/A"
					print(f"Output (last 100 chars): ...{output[-100:]}")

			for idx, output in enumerate(outputs):
				flag = bool(flags[idx]) if idx < len(flags) else False
				program_text = programs[idx] if idx < len(programs) else ""
				bucket = bucket_correctness(program_text, flag)
				per_difficulty_correctness[difficulty][bucket] += 1

	print({
		"total_records": total_records,
		"records_with_empty_program": len(ids_with_empty),
		"empty_program_entries": empty_program_entries,
	})
	print("ids_with_empty_programs:", ids_with_empty)

	if difficulty_counts:
		total_problems = sum(difficulty_counts.values())
		print("Problem counts by difficulty:")
		for diff in sorted(difficulty_counts.keys()):
			count = difficulty_counts[diff]
			pct = (count / total_problems) * 100 if total_problems else 0
			print(f"  {diff}: {count} ({pct:.1f}%)")

		print("Output correctness by difficulty:")
		for diff in sorted(per_difficulty_correctness.keys()):
			counts = per_difficulty_correctness[diff]
			total = sum(counts.values())
			print(
				f"  {diff}: correct={counts['correct']}, no_program={counts['no_program']}, "
				f"tests_failed={counts['tests_failed']} (total outputs={total})"
			)

	plot_path = output_dir / f"{args.save_prefix}_distributions.png"
	plot_distributions(
		difficulty_counts=difficulty_counts,
		per_difficulty_correctness=per_difficulty_correctness,
		save_path=plot_path,
	)
	print(f"Saved plots to {plot_path}")


if __name__ == "__main__":
	main()