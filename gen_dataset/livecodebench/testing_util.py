from __future__ import annotations

"""
This is a lightweight adaptation of lcb_runner.evaluation.testing_util that
supports the two evaluation modes used in the LCB code-generation split:
  * call-based (LeetCode-style) via a named function
  * standard-input problems that read from stdin and write to stdout

The implementation is not a security sandbox; do not run untrusted code outside of a controlled environment.
"""

import ast
import faulthandler
import json
import multiprocessing
import os
import platform
import signal
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from io import StringIO
from types import ModuleType
from typing import Any, Callable, List, Optional, Tuple
from unittest.mock import patch, mock_open



class TimeoutException(Exception):
	pass


def _timeout_handler(signum, frame):  # pragma: no cover - signal handler
	raise TimeoutException


class CodeType(Enum):
	CALL_BASED = 0
	STDIN = 1


IMPORT_STUB = (
	"from string import *\n"
	"from re import *\n"
	"from datetime import *\n"
	"from collections import *\n"
	"from heapq import *\n"
	"from bisect import *\n"
	"from copy import *\n"
	"from math import *\n"
	"from random import *\n"
	"from statistics import *\n"
	"from itertools import *\n"
	"from functools import *\n"
	"from operator import *\n"
	"from io import *\n"
	"from sys import *\n"
	"from json import *\n"
	"from builtins import *\n"
	"from typing import *\n"
	"import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\n"
)


def reliability_guard(maximum_memory_bytes: Optional[int] = None) -> None:
	if maximum_memory_bytes is not None:
		import resource

		resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
		resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
		if platform.uname().system != "Darwin":
			resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

	faulthandler.disable()
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


def _truncate(val: Any, length: int = 300) -> str:
	s = val if isinstance(val, str) else str(val)
	if len(s) <= length:
		return s
	return s[: length // 2] + "...(truncated)..." + s[-length // 2 :]


def _convert_line_to_decimals(line: str) -> Tuple[bool, List[Decimal]]:
	try:
		return True, [Decimal(elem) for elem in line.split()]
	except Exception:
		return False, []


def _get_stripped_lines(val: str) -> List[str]:
	val = val.strip()
	return [line.strip() for line in val.split("\n")]


def _compile_code(code: str, timeout: int) -> Optional[Any]:
	signal.alarm(timeout)
	try:
		tmp_sol = ModuleType("tmp_sol", "")
		exec(code, tmp_sol.__dict__)
		if "class Solution" in code:
			compiled = tmp_sol.Solution()
		else:
			compiled = tmp_sol
		return compiled
	finally:
		signal.alarm(0)


def _get_function(compiled_sol: Any, fn_name: str) -> Optional[Callable]:
	return getattr(compiled_sol, fn_name, None)


def _clean_if_main(code: str) -> str:
	try:
		astree = ast.parse(code)
		last_block = astree.body[-1]
		if isinstance(last_block, ast.If):
			condition = last_block.test
			if ast.unparse(condition).strip() == "__name__ == '__main__'":
				code = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore[arg-type]
	except Exception:
		return code
	return code


def _wrap_as_function(code: str) -> str:
	try:
		import_stmts: List[ast.stmt] = []
		body: List[ast.stmt] = []
		astree = ast.parse(code)
		for stmt in astree.body:
			if isinstance(stmt, (ast.Import, ast.ImportFrom)):
				import_stmts.append(stmt)
			else:
				body.append(stmt)
		fn_ast = ast.FunctionDef(
			name="wrapped_function",
			args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
			body=body,
			decorator_list=[],
			lineno=-1,
		)
		imports_src = "\n".join(ast.unparse(stmt) for stmt in import_stmts)
		return IMPORT_STUB + "\n" + imports_src + "\n" + ast.unparse(fn_ast)
	except Exception:
		return code


def _call_method(method: Callable, inputs: str):
	inputs_line_iterator = iter(inputs.split("\n"))
	mock_stdin = StringIO(inputs)

	@patch("builtins.open", mock_open(read_data=inputs))
	@patch("sys.stdin", mock_stdin)
	@patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
	@patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
	@patch("sys.stdin.read", lambda *args: inputs)
	def _inner_call(_method):
		return _method()

	return _inner_call(method)


def _grade_call_based(code: str, all_inputs: List[str], all_outputs: List[str], fn_name: str, timeout: int):
	code = IMPORT_STUB + "\n\n" + code
	compiled_sol = _compile_code(code, timeout)
	if compiled_sol is None:
		return [-4], {"error_message": "compile failed", "error_code": -4}
	method = _get_function(compiled_sol, fn_name)
	if method is None:
		return [-4], {"error_message": "function not found", "error_code": -4}

	parsed_inputs = [[json.loads(line) for line in inp.split("\n")] for inp in all_inputs]
	parsed_outputs = [json.loads(out) for out in all_outputs]
	results = []
	total_exec = 0.0
	for gt_inp, gt_out in zip(parsed_inputs, parsed_outputs):
		signal.alarm(timeout)
		faulthandler.enable()
		try:
			start = time.time()
			prediction = method(*gt_inp)
			signal.alarm(0)
			total_exec += time.time() - start
			if isinstance(prediction, tuple):
				prediction = list(prediction)
			if isinstance(gt_out, tuple):
				gt_out = list(gt_out)
			if isinstance(prediction, list) and isinstance(gt_out, list):
				results.append(prediction == gt_out)
				continue
			try:
				success_pred, dec_pred = _convert_line_to_decimals(str(prediction))
				success_gt, dec_gt = _convert_line_to_decimals(str(gt_out))
				if success_pred and success_gt:
					results.append(dec_pred == dec_gt)
					continue
			except Exception:
				pass
			results.append(prediction == gt_out)
		except TimeoutException:
			results.append(-3)
			return results, {"error_code": -3, "error_message": "Time Limit Exceeded"}
		except Exception as exc:
			results.append(-4)
			return results, {"error_code": -4, "error_message": repr(exc)}
		finally:
			signal.alarm(0)
			faulthandler.disable()
	return results, {"execution_time": total_exec}


def _grade_stdio(code: str, all_inputs: List[str], all_outputs: List[str], timeout: int):
	code = _clean_if_main(code)
	code = _wrap_as_function(code)
	compiled_sol = _compile_code(code, timeout)
	if compiled_sol is None:
		return [-4], {"error_message": "compile failed", "error_code": -4}
	method = _get_function(compiled_sol, "wrapped_function")
	if method is None:
		return [-4], {"error_message": "wrapped_function missing", "error_code": -4}

	results = []
	for gt_inp, gt_out in zip(all_inputs, all_outputs):
		signal.alarm(timeout)
		faulthandler.enable()
		prediction = ""
		try:
			with patch("sys.stdout", new_callable=StringIO) as fake_out:
				_call_method(method, gt_inp)
				prediction = fake_out.getvalue()
		finally:
			signal.alarm(0)
			faulthandler.disable()

			pred_lines = _get_stripped_lines(prediction)
			gt_lines = _get_stripped_lines(gt_out)
			if len(pred_lines) != len(gt_lines):
				results.append(-2)
				return results, {
					"error_code": -2,
					"error_message": "Wrong answer: mismatched output length",
					"output": _truncate(prediction),
					"expected": _truncate(gt_out),
				}
			all_ok = True
			for pl, gl in zip(pred_lines, gt_lines):
				if pl == gl:
					continue
				success_pred, dec_pred = _convert_line_to_decimals(pl)
				success_gt, dec_gt = _convert_line_to_decimals(gl)
				if success_pred and success_gt:
					all_ok = all_ok and dec_pred == dec_gt
				else:
					all_ok = False
			if not all_ok:
				results.append(-2)
				return results, {
					"error_code": -2,
					"error_message": "Wrong Answer",
					"output": _truncate(prediction),
					"expected": _truncate(gt_out),
				}
			results.append(True)
	return results, {"execution_time": None}


@dataclass
class RunResult:
	results: List[Any]
	metadata: dict


def _run_test_impl(sample: dict, test: Optional[str], debug: bool, timeout: int) -> Tuple[List[Any], dict]:
	signal.signal(signal.SIGALRM, _timeout_handler)
	reliability_guard()

	if test is None:
		return [-4], {"error": "no test code provided", "error_code": -4}

	try:
		in_outs = json.loads(sample["input_output"])
	except Exception as exc:
		return [-4], {"error": f"invalid input_output: {exc}", "error_code": -4}

	inputs = in_outs.get("inputs", [])
	outputs = in_outs.get("outputs", [])
	fn_name = in_outs.get("fn_name")

	if fn_name is None:
		code_type = CodeType.STDIN
	else:
		code_type = CodeType.CALL_BASED

	if code_type == CodeType.CALL_BASED:
		return _grade_call_based(test, inputs, outputs, fn_name, timeout)
	return _grade_stdio(test, inputs, outputs, timeout)


def _run_test_subprocess(sample: dict, test: Optional[str], debug: bool, timeout: int, conn) -> None:
	os_attr_names = [
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
	]
	original_os_attrs = {name: getattr(os, name, None) for name in os_attr_names}
	try:
		results, metadata = _run_test_impl(sample, test, debug, timeout)
	except Exception as exc:
		results = [-4]
		metadata = {"error_code": -4, "error_message": repr(exc)}
	finally:
		for name, original in original_os_attrs.items():
			if original is not None and getattr(os, name, None) is None:
				setattr(os, name, original)
	try:
		conn.send((results, metadata))
	finally:
		conn.close()


def _compute_hard_timeout(sample: dict, timeout: int) -> int:
	try:
		in_outs = json.loads(sample.get("input_output", "{}"))
		inputs = in_outs.get("inputs", [])
		num_tests = len(inputs) if isinstance(inputs, list) else 1
		total = max(1, num_tests) * max(1, int(timeout)) + 5
		return max(30, total)
	except Exception:
		return max(30, max(1, int(timeout)) * 10)


def run_test(sample: dict, test: Optional[str] = None, debug: bool = False, timeout: int = 6) -> Tuple[List[Any], dict]:
	"""Execute generated code against sample tests.

	Returns (results, metadata) where results is a list of per-test booleans or
	negative error codes.
	"""
	ctx = multiprocessing.get_context("spawn")
	parent_conn, child_conn = ctx.Pipe(duplex=False)
	proc = ctx.Process(target=_run_test_subprocess, args=(sample, test, debug, timeout, child_conn))
	proc.start()
	child_conn.close()

	hard_timeout = _compute_hard_timeout(sample, timeout)
	proc.join(hard_timeout)
	if proc.is_alive():
		proc.terminate()
		proc.join(5)
		parent_conn.close()
		return [-3], {"error_code": -3, "error_message": "subprocess timed out"}

	if parent_conn.poll():
		results, metadata = parent_conn.recv()
		parent_conn.close()
		return results, metadata

	parent_conn.close()
	return [-4], {
		"error_code": -4,
		"error_message": f"subprocess crashed (exitcode={proc.exitcode})",
	}
