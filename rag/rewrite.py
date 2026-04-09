from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import requests

from .prompt import format_chat_history


class OllamaRewriter:
    """
    Retrieval-only query refinement + multi-query generator using Ollama.

    LOCKED RULE:
    - refined/re-written query is used ONLY for retrieval
    - final answer LLM does NOT see the refined query
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: int = 120,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.model = (
            model
            or os.getenv("OLLAMA_REWRITE_MODEL")
            or os.getenv("OLLAMA_CHAT_MODEL")
            or "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
        )
        self.timeout_s = timeout_s

    def _generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 96,
    ) -> str:
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
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    def refine_query(
        self,
        raw_query: str,
        chat_history: List[Tuple[str, str]],
        memory_max_chars: int = 6000,
    ) -> str:
        """
        Returns a refined query for retrieval purposes.
        If refinement isn't needed, returns raw_query.
        """
        memory = format_chat_history(chat_history, max_chars=memory_max_chars)

        prompt = f"""You refine user search queries for document retrieval.

Conversation History:
{memory if memory.strip() else "(none)"}

Raw User Query:
{raw_query}

Instructions:
- Use conversation history ONLY to add missing context/keywords that the user is clearly referring to.
- Keep the refined query concise but information-rich.
- If the raw query already contains enough context, output it unchanged.
- Output ONLY the refined query text. No extra words, no labels, no quotes.

Refined Query:
"""
        out = self._generate(prompt, temperature=0.2, top_p=0.9, top_k=40, num_predict=80)

        if not out:
            return raw_query.strip()

        out = out.strip().strip('"').strip()
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return (lines[0] if lines else raw_query).strip()

    def multi_query(self, retrieval_query: str, nq: int) -> List[str]:
        """
        Generate nq retrieval queries.
        - nq <= 1 -> [retrieval_query]
        - nq == 3 -> generate 3 diverse retrieval queries (JSON array)
        """
        if nq <= 1:
            return [retrieval_query.strip()]

        prompt = f"""Generate {nq} diverse search queries for retrieving documents.

Base Query:
{retrieval_query}

Rules:
- Output a JSON array of exactly {nq} strings.
- Each string should be a good retrieval query (keywords + short phrase).
- Output ONLY JSON (no commentary).

JSON:
"""
        out = self._generate(prompt, temperature=0.35, top_p=0.95, top_k=50, num_predict=140).strip()

        try:
            arr = json.loads(out)
            if isinstance(arr, list):
                qs = [str(x).strip() for x in arr if str(x).strip()]
                if len(qs) >= nq:
                    return qs[:nq]
        except Exception:
            pass

        lines = [ln.strip("-• \t").strip() for ln in out.splitlines() if ln.strip()]
        if len(lines) >= nq:
            return lines[:nq]

        return [retrieval_query.strip()] * nq