from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGConfig:
    """
    Baseline configuration for one RAG run.

    IMPORTANT RULE (locked):
    - refined/re-written query is used ONLY for retrieval
    - raw user query + memory + top-k chunks go to the answer LLM
    """

    # Query rewrite / multi-query (retrieval-only)
    rewrite: bool = True
    nq: int = 3  # number of retrieval queries (1 or 3 for now)

    # Candidate retrieval sizes (per retriever, per retrieval query)
    bm25_kcand: int = 50
    dense_kcand: int = 50

    # Hybrid weighting: x*bm25_norm + y*dense_norm where x+y=1
    x_bm25: float = 0.5
    y_dense: float = 0.5

    # Reranking
    rerank: bool = True
    rerank_topR: int = 30

    # Final context size
    k_final: int = 10

    # Memory handling (for final answer prompt)
    memory_max_chars: int = 6000  # simple cap; tokens computed later if needed

    # Allow retrieval skip (baseline will keep retrieval on; RL later may set to 0)
    allow_retrieve_zero: bool = False

    # Safety / sanity
    min_x_y_tol: float = 1e-6

    def __post_init__(self) -> None:
        if self.nq not in (1, 3):
            raise ValueError(f"nq must be 1 or 3 for checkpoint baseline, got {self.nq}")

        if self.bm25_kcand <= 0 or self.dense_kcand <= 0:
            raise ValueError("bm25_kcand and dense_kcand must be > 0")

        if self.rerank_topR <= 0:
            raise ValueError("rerank_topR must be > 0")

        if self.k_final <= 0:
            raise ValueError("k_final must be > 0")

        if not (0.0 <= self.x_bm25 <= 1.0) or not (0.0 <= self.y_dense <= 1.0):
            raise ValueError("x_bm25 and y_dense must be in [0, 1]")

        if abs((self.x_bm25 + self.y_dense) - 1.0) > self.min_x_y_tol:
            # auto-fix if tiny float drift, else raise
            s = self.x_bm25 + self.y_dense
            if s > 0:
                self.x_bm25 = self.x_bm25 / s
                self.y_dense = self.y_dense / s
            else:
                raise ValueError("x_bm25 + y_dense must sum to 1")

        # If rerank is off, rerank_topR doesn't matter but keep it valid
        if not self.rerank:
            self.rerank_topR = max(self.rerank_topR, self.k_final)

        # Make sure rerank pool >= final k
        if self.rerank_topR < self.k_final:
            self.rerank_topR = self.k_final

    def to_dict(self) -> dict:
        return {
            "rewrite": self.rewrite,
            "nq": self.nq,
            "bm25_kcand": self.bm25_kcand,
            "dense_kcand": self.dense_kcand,
            "x_bm25": self.x_bm25,
            "y_dense": self.y_dense,
            "rerank": self.rerank,
            "rerank_topR": self.rerank_topR,
            "k_final": self.k_final,
            "memory_max_chars": self.memory_max_chars,
            "allow_retrieve_zero": self.allow_retrieve_zero,
        }