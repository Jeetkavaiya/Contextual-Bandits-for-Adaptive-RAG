from __future__ import annotations

import math
import re
from typing import Dict, List

from rag.index_bm25 import BM25Index
from rag.index_dense import DenseIndex


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _has_any(q: str, words: List[str]) -> int:
    ql = q.lower()
    return 1 if any(w in ql for w in words) else 0


STATE_KEYS = [
    "len_words",
    "len_chars",
    "has_list",
    "has_steps",
    "has_compare",
    "has_why",
    "has_math",
    "bm25_top",
    "bm25_gap",
    "dense_top",
    "dense_gap",
]


def extract_state(
    raw_query: str,
    bm25: BM25Index,
    dense: DenseIndex,
) -> Dict[str, float]:
    """
    State = cheap query features + cheap retrieval confidence probe (k=5).
    Uses RAW query only (rewrite is an action later).
    """
    q = raw_query.strip()

    words = _WORD_RE.findall(q)
    len_words = len(words)
    len_chars = len(q)

    has_list = _has_any(q, ["list", "all", "topics", "types", "examples"])
    has_steps = _has_any(q, ["steps", "procedure", "process", "how to", "checklist"])
    has_compare = _has_any(q, ["compare", "difference", "vs", "versus"])
    has_why = _has_any(q, ["why", "reason"])
    has_math = 1 if re.search(r"\d+\s*[\+\-\*/]\s*\d+", q) else 0

    # Probe retrieval
    bm_hits = bm25.search(q, k=5)
    de_hits = dense.search(q, k=5)

    bm_scores = [h.score for h in bm_hits]
    de_scores = [h.score for h in de_hits]

    bm_top = bm_scores[0] if bm_scores else 0.0
    de_top = de_scores[0] if de_scores else 0.0

    bm_tail = bm_scores[1:] if len(bm_scores) > 1 else []
    de_tail = de_scores[1:] if len(de_scores) > 1 else []

    bm_gap = (bm_top - (sum(bm_tail) / len(bm_tail))) if bm_tail else bm_top
    de_gap = (de_top - (sum(de_tail) / len(de_tail))) if de_tail else de_top

    return {
        "len_words": float(len_words),
        "len_chars": float(len_chars),
        "has_list": float(has_list),
        "has_steps": float(has_steps),
        "has_compare": float(has_compare),
        "has_why": float(has_why),
        "has_math": float(has_math),
        "bm25_top": float(bm_top),
        "bm25_gap": float(bm_gap),
        "dense_top": float(de_top),
        "dense_gap": float(de_gap),
    }


def state_to_vector(state: Dict[str, float]) -> List[float]:
    """
    Convert dict -> vector in stable order (STATE_KEYS).
    """
    return [float(state.get(k, 0.0)) for k in STATE_KEYS]