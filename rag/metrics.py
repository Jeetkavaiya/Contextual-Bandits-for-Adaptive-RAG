from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

from .types import Chunk
from .llm import OllamaLLM


@dataclass
class EvalItem:
    qid: str
    raw_query: str
    chat_history: List[Tuple[str, str]] = field(default_factory=list)

    gold_answer: Optional[str] = None
    gold_support_chunk_ids: List[str] = field(default_factory=list)

    domain: str = "in_domain"  # in_domain / out_of_domain


def save_evalset_jsonl(items: List[EvalItem], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False))
            f.write("\n")


def load_evalset_jsonl(path: str) -> List[EvalItem]:
    items: List[EvalItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(
                EvalItem(
                    qid=obj["qid"],
                    raw_query=obj["raw_query"],
                    chat_history=[tuple(x) for x in obj.get("chat_history", [])],
                    gold_answer=obj.get("gold_answer", None),
                    gold_support_chunk_ids=list(obj.get("gold_support_chunk_ids", [])),
                    domain=obj.get("domain", "in_domain"),
                )
            )
    return items


def default_out_of_domain_items() -> List[EvalItem]:
    qs = [
        ("ood_001", "What is the capital of India?"),
        ("ood_002", "Solve: 37 * 19"),
        ("ood_003", "What is the boiling point of water in Celsius at sea level?"),
        ("ood_004", "Who wrote 'Pride and Prejudice'?"),
        ("ood_005", "What is 15% of 260?"),
        ("ood_006", "What is the capital of New York state?"),
        ("ood_007", "Simplify: (x^2 * x^3)"),
        ("ood_008", "What is the largest planet in our solar system?"),
        ("ood_009", "Convert 5 kilometers to meters."),
        ("ood_010", "What year did World War II end?"),
    ]
    items = []
    for qid, q in qs:
        items.append(EvalItem(qid=qid, raw_query=q, domain="out_of_domain"))
    return items


def generate_in_domain_items_from_chunks(
    chunks: List[Chunk],
    n: int,
    seed: int = 7,
    model: Optional[str] = None,
) -> List[EvalItem]:
    """
    Generates (question, short gold answer) pairs from randomly sampled chunks.
    Each item gets gold_support_chunk_ids = [the source chunk_id] (simple but strong for Recall@k).

    This runs locally using Ollama (no APIs).
    """
    rng = random.Random(seed)
    llm = OllamaLLM(model=model)

    sampled = rng.sample(chunks, k=min(n, len(chunks)))
    items: List[EvalItem] = []

    for i, ch in enumerate(sampled):
        prompt = f"""Create one question and a short answer based ONLY on the text below.

Text:
{ch.text}

Rules:
- The question must be answerable from the text.
- The answer must be short (1-2 sentences or a small list).
- Output STRICT JSON only:
{{
  "question": "...",
  "answer": "..."
}}
"""
        raw, _, _ = llm.generate(prompt, temperature=0.2, top_p=0.9, top_k=40, num_predict=200)

        # parse JSON
        q = None
        a = None
        try:
            obj = json.loads(raw.strip())
            q = str(obj.get("question", "")).strip()
            a = str(obj.get("answer", "")).strip()
        except Exception:
            # fallback: if parse fails, skip
            continue

        if not q or not a:
            continue

        items.append(
            EvalItem(
                qid=f"id_{i:03d}",
                raw_query=q,
                gold_answer=a,
                gold_support_chunk_ids=[ch.chunk_id],
                domain="in_domain",
            )
        )

    return items