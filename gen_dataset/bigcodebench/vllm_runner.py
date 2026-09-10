from __future__ import annotations

import logging
import gc
from typing import Iterable, List, Literal, Dict, Any

try:
	from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover
	raise ImportError("vllm is required for VLLMRunner") from exc

LOGGER = logging.getLogger(__name__)


class VLLMRunner:
	def __init__(
		self,
		*,
		model_name: str,
		tokenizer_name: str,
		n: int,
		temperature: float,
		top_p: float,
		top_k: int,
		max_tokens: int,
		max_model_len: int,
		gpu_memory_utilization: float,
		tensor_parallel_size: int = 1,
		dtype: Literal["auto", "bfloat16", "float32"] = "bfloat16",
		repetition_penalty: float = 1,
		enable_prefix_caching: bool = False,
		trust_remote_code: bool = False,
	) -> None:
		self.sampling_params = SamplingParams(
			n=n,
			max_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			top_k=top_k,
			repetition_penalty=repetition_penalty,
		)

		LOGGER.info("Initializing vLLM model %s", model_name)
		self.llm = LLM(
			model=model_name,
			tokenizer=tokenizer_name,
			tensor_parallel_size=tensor_parallel_size,
			dtype=dtype,
			max_model_len=max_model_len,
			gpu_memory_utilization=gpu_memory_utilization,
			enable_prefix_caching=enable_prefix_caching,
			trust_remote_code=trust_remote_code,
		)

	def shutdown(self) -> None:
		llm = getattr(self, "llm", None)
		if llm is None:
			return
		try:
			shutdown_fn = getattr(llm, "shutdown", None)
			if callable(shutdown_fn):
				shutdown_fn()
		finally:
			self.llm = None
			gc.collect()

	def generate(self, prompts: Iterable[str]) -> List[List[Dict[str, Any]]]:
		prompts_list = list(prompts)
		if not prompts_list:
			return []
		if self.llm is None:
			raise RuntimeError("VLLMRunner has been shut down")
		outputs = self.llm.generate(prompts_list, self.sampling_params)
		result: List[List[Dict[str, Any]]] = []
		for output in outputs:
			cand_list: List[Dict[str, Any]] = []
			for cand in output.outputs:
				cand_list.append({
					"text": cand.text,
					"token_ids": getattr(cand, "token_ids", None),
				})
			result.append(cand_list)
		return result
