from __future__ import annotations

from typing import Optional

from .problems import CodeGenerationProblem

DEFAULT_SYSTEM_PROMPT = (
	"You are an expert Python programmer. You will be given a question (problem specification) "
	"and will generate a correct Python program that matches the specification and passes all tests."
)


def build_user_prompt(problem: CodeGenerationProblem) -> str:
	prompt_lines = [f"### Question:\n{problem.question_content}\n"]
	if problem.starter_code:
		prompt_lines.append("### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n")
		prompt_lines.append(f"```python\n{problem.starter_code}\n```\n")
	else:
		prompt_lines.append(
			"### Format: Read the inputs from stdin, solve the problem, and write the answer to stdout. "
			"Enclose your code within delimiters as follows.\n"
		)
		prompt_lines.append("```python\n# YOUR CODE HERE\n```\n")
	prompt_lines.append("### Answer: (use the provided format with backticks)\n\n")
	return "".join(prompt_lines)


def build_chat_messages(problem: CodeGenerationProblem, system_prompt: Optional[str]) -> list[dict[str, str]]:
	messages = []
	if system_prompt:
		messages.append({"role": "system", "content": system_prompt})
	messages.append({"role": "user", "content": build_user_prompt(problem)})
	return messages
