from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from .llm import OllamaLLM


@dataclass
class JudgeResult:
    correctness: int  # 0/1/2
    faithfulness: int  # 0/1
    notes: str = ""


class OllamaJudge:
    """
    LLM-as-judge using Ollama.

    Returns:
    - correctness: 0 (wrong), 1 (partial), 2 (correct)
    - faithfulness: 0 (not supported by context), 1 (supported / doesn't contradict context)

    For out-of-domain questions: context may be irrelevant, so faithfulness checks that the answer
    does NOT claim to use context incorrectly.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        # Use a dedicated judge model if you want; else fallback to same chat model
        self.model = model or os.getenv("OLLAMA_JUDGE_MODEL") or os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_ANSWER_MODEL")
        self.llm = OllamaLLM(model=self.model)

    @staticmethod
    def _safe_json_extract(text: str) -> Optional[dict]:
        text = text.strip()
        # Try direct JSON parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Try to locate first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                obj = json.loads(snippet)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

    def score(
        self,
        question: str,
        predicted_answer: str,
        context: str,
        gold_answer: Optional[str] = None,
    ) -> JudgeResult:
        gold = gold_answer if gold_answer is not None else "(none)"

        prompt = f"""You are grading a QA system.

Question:
{question}

Predicted Answer:
{predicted_answer}

Gold Answer (if available):
{gold}

Context Provided to the model:
{context if context.strip() else "(no context)"}

Rubric:
1) correctness (integer):
   2 = fully correct
   1 = partially correct / missing detail
   0 = incorrect
2) faithfulness (integer):
   1 = answer is supported by the provided context OR the answer correctly uses general knowledge when context is irrelevant
   0 = answer claims facts not supported by context when it should, or contradicts context, or hallucinates citations

Output STRICT JSON only:
{{
  "correctness": 0|1|2,
  "faithfulness": 0|1,
  "notes": "one short sentence"
}}
"""
        raw, _, _ = self.llm.generate(prompt, temperature=0.0, top_p=1.0, top_k=1, num_predict=180)

        obj = self._safe_json_extract(raw)
        if not obj:
            # fallback if judge output is messy
            return JudgeResult(correctness=0, faithfulness=0, notes="judge_parse_failed")

        c = obj.get("correctness", 0)
        f = obj.get("faithfulness", 0)
        notes = str(obj.get("notes", "")).strip()

        try:
            c = int(c)
        except Exception:
            c = 0
        try:
            f = int(f)
        except Exception:
            f = 0

        c = 0 if c < 0 else 2 if c > 2 else c
        f = 0 if f < 0 else 1 if f > 1 else f

        return JudgeResult(correctness=c, faithfulness=f, notes=notes)