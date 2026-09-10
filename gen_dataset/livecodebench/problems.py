from __future__ import annotations

import base64
import json
import pickle
import zlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional

from datasets import load_dataset


class Platform(str, Enum):
	LEETCODE = "leetcode"
	CODEFORCES = "codeforces"
	ATCODER = "atcoder"


class Difficulty(str, Enum):
	EASY = "easy"
	MEDIUM = "medium"
	HARD = "hard"


class TestType(str, Enum):
	STDIN = "stdin"
	FUNCTIONAL = "functional"


@dataclass
class TestCase:
	input: str
	output: str
	testtype: TestType

	def __post_init__(self) -> None:
		self.testtype = TestType(self.testtype)


@dataclass
class CodeGenerationProblem:
	question_title: Any
	question_content: Any
	platform: Any
	question_id: Any
	contest_id: Any
	contest_date: Any
	starter_code: Any
	difficulty: Any
	public_test_cases: Any
	private_test_cases: Any
	metadata: Any

	def __post_init__(self) -> None:
		self.platform = Platform(self.platform)
		self.difficulty = Difficulty(self.difficulty)
		self.contest_date = datetime.fromisoformat(self.contest_date)

		public = json.loads(self.public_test_cases)
		self.public_test_cases = [TestCase(**case) for case in public]

		private_raw = None
		# First attempt: JSON directly.
		try:
			direct = json.loads(self.private_test_cases)
			if isinstance(direct, list):
				private_raw = direct
			elif isinstance(direct, str):
				private_raw = json.loads(direct)
		except Exception:
			private_raw = None

		# Second attempt: base64 + zlib + pickle (dataset default).
		if private_raw is None:
			decoded = base64.b64decode(self.private_test_cases.encode("utf-8"))
			try:
				inflated = zlib.decompress(decoded)
			except Exception:
				inflated = decoded
			try:
				pickled_obj = pickle.loads(inflated)
				if isinstance(pickled_obj, str):
					private_raw = json.loads(pickled_obj)
				elif isinstance(pickled_obj, list):
					private_raw = pickled_obj
			except Exception:
				private_raw = None

		if not isinstance(private_raw, list):
			raise ValueError("Unable to parse private_test_cases into a list")
		self.private_test_cases = [TestCase(**case) for case in private_raw]

		self.metadata = json.loads(self.metadata)

	def build_eval_sample(self) -> dict:
		inputs = [t.input for t in self.public_test_cases + self.private_test_cases]
		outputs = [t.output for t in self.public_test_cases + self.private_test_cases]
		return {
			"input_output": json.dumps(
				{
					"inputs": inputs,
					"outputs": outputs,
					"fn_name": self.metadata.get("func_name", None),
				}
			)
		}


def load_code_generation_dataset(
	release_version: str = "release_latest",
	*,
	start_date: Optional[str] = None,
	end_date: Optional[str] = None,
	use_lite_split: bool = True,
) -> list[CodeGenerationProblem]:
	ds_name = "livecodebench/code_generation_lite" if use_lite_split else "livecodebench/code_generation"
	dataset = load_dataset(ds_name, split="test", version_tag=release_version, trust_remote_code=True)

	problems = [CodeGenerationProblem(**item) for item in dataset]  # type: ignore[arg-type]
	if start_date:
		start_dt = datetime.strptime(start_date, "%Y-%m-%d")
		problems = [p for p in problems if p.contest_date >= start_dt]
	if end_date:
		end_dt = datetime.strptime(end_date, "%Y-%m-%d")
		problems = [p for p in problems if p.contest_date <= end_dt]
	return sorted(problems, key=lambda p: p.question_id)
