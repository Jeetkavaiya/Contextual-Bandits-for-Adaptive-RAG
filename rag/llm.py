from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import requests

from .types import TokenUsage


class OllamaLLM:
    """
    Wrapper for Ollama /api/generate.

    Returns:
      (answer_text, TokenUsage, gen_time_seconds)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 600,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = (
            model
            or os.getenv("OLLAMA_ANSWER_MODEL")
            or os.getenv("OLLAMA_CHAT_MODEL")
            or "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
        )
        self.timeout_s = timeout_s

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 256,
    ) -> Tuple[str, TokenUsage, float]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": num_predict,
            },
        }

        t0 = time.perf_counter()
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        t1 = time.perf_counter()

        answer = (data.get("response") or "").strip()

        prompt_tokens = data.get("prompt_eval_count", None)
        completion_tokens = data.get("eval_count", None)

        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        usage = TokenUsage(
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total_tokens,
        )

        return answer, usage, (t1 - t0)