from __future__ import annotations

import re
from typing import Dict, List

from .types import Chunk, Hit

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


class BM25Index:
    """
    BM25 over chunk texts using rank-bm25.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except Exception as e:
            raise ImportError("Missing dependency rank-bm25. Install: pip install rank-bm25") from e

        self._chunks = list(chunks)
        tokenized = [tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 10) -> List[Hit]:
        if self._bm25 is None:
            raise RuntimeError("BM25Index not built. Call build(chunks) first.")
        if k <= 0:
            return []

        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)  # type: ignore

        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits: List[Hit] = []
        for i in top_idx:
            c = self._chunks[i]
            s = float(scores[i])
            hits.append(
                Hit(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    retriever="bm25",
                    score=s,
                    bm25_score=s,
                    meta=c.meta,
                )
            )
        return hits