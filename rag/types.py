from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


RetrieverName = Literal["bm25", "dense", "hybrid", "rerank"]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    text: str
    retriever: RetrieverName
    score: float

    # Optional component scores for debugging/analysis
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    bm25_norm: Optional[float] = None
    dense_norm: Optional[float] = None

    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingInfo:
    total_s: float = 0.0
    rewrite_s: float = 0.0
    retrieve_s: float = 0.0
    hybrid_s: float = 0.0
    rerank_s: float = 0.0
    build_prompt_s: float = 0.0
    gen_s: float = 0.0


@dataclass
class TokenUsage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    context_tokens: Optional[int] = None
    memory_tokens: Optional[int] = None


@dataclass
class RAGResult:
    raw_query: str
    retrieval_query: str
    retrieval_queries: List[str]

    answer: str

    # Final context passed to generator
    context_chunks: List[Chunk]
    context_hits: List[Hit]

    # Useful for analysis/debug
    timings: TimingInfo
    tokens: TokenUsage
    meta: Dict[str, Any] = field(default_factory=dict)