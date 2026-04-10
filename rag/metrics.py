from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .evalset import EvalItem
from .judge import OllamaJudge, JudgeResult
from .types import RAGResult


def evidence_recall_at_k(retrieved_chunk_ids: List[str], gold_chunk_ids: List[str]) -> int:
    """
    1 if any gold support chunk id appears in retrieved context chunk ids, else 0.
    """
    if not gold_chunk_ids:
        return 0
    got = set(retrieved_chunk_ids)
    return 1 if any(cid in got for cid in gold_chunk_ids) else 0


def build_context_string_for_judge(res: RAGResult, max_chars: int = 6000) -> str:
    """
    Compact string of the final context that was passed to the answer LLM.
    Used only for evaluation.
    """
    parts: List[str] = []
    for c in res.context_chunks:
        page = c.meta.get("page", None)
        hdr = f"[{c.chunk_id}]"
        if page is not None:
            hdr += f" (page {page})"
        parts.append(f"{hdr}\n{c.text}")
    ctx = "\n\n".join(parts).strip()
    return ctx[:max_chars]


@dataclass
class MetricsRow:
    qid: str
    domain: str

    correctness: Optional[int] = None      # 0/1/2
    faithfulness: Optional[int] = None     # 0/1
    judge_notes: str = ""

    evidence_recall_at_k: Optional[int] = None  # 0/1 (in-domain only)

    total_time_s: float = 0.0
    rewrite_s: float = 0.0
    retrieve_s: float = 0.0
    hybrid_s: float = 0.0
    rerank_s: float = 0.0
    build_prompt_s: float = 0.0
    gen_s: float = 0.0

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    context_tokens_est: Optional[int] = None
    memory_tokens_est: Optional[int] = None

    k_final: int = 0
    nq: int = 0
    rerank_enabled: int = 0
    rewrite_enabled: int = 0
    x_bm25: float = 0.0
    y_dense: float = 0.0


def score_one(item: EvalItem, res: RAGResult, judge: Optional[OllamaJudge] = None) -> MetricsRow:
    """
    Turns one RAGResult into one metrics row using Ollama judge.
    """
    judge = judge or OllamaJudge()

    retrieved_ids = [c.chunk_id for c in res.context_chunks]
    ctx = build_context_string_for_judge(res)

    jr: JudgeResult = judge.score(
        question=item.raw_query,
        predicted_answer=res.answer,
        context=ctx,
        gold_answer=item.gold_answer,
    )

    cfg = res.meta.get("config", {}) if isinstance(res.meta, dict) else {}

    row = MetricsRow(
        qid=item.qid,
        domain=item.domain,
        correctness=jr.correctness,
        faithfulness=jr.faithfulness,
        judge_notes=jr.notes,
        evidence_recall_at_k=evidence_recall_at_k(retrieved_ids, item.gold_support_chunk_ids),
        total_time_s=float(res.timings.total_s),
        rewrite_s=float(res.timings.rewrite_s),
        retrieve_s=float(res.timings.retrieve_s),
        hybrid_s=float(res.timings.hybrid_s),
        rerank_s=float(res.timings.rerank_s),
        build_prompt_s=float(res.timings.build_prompt_s),
        gen_s=float(res.timings.gen_s),
        prompt_tokens=res.tokens.prompt_tokens,
        completion_tokens=res.tokens.completion_tokens,
        total_tokens=res.tokens.total_tokens,
        context_tokens_est=res.tokens.context_tokens,
        memory_tokens_est=res.tokens.memory_tokens,
        k_final=int(cfg.get("k_final", 0) or 0),
        nq=int(cfg.get("nq", 0) or 0),
        rerank_enabled=1 if cfg.get("rerank", False) else 0,
        rewrite_enabled=1 if cfg.get("rewrite", False) else 0,
        x_bm25=float(cfg.get("x_bm25", 0.0) or 0.0),
        y_dense=float(cfg.get("y_dense", 0.0) or 0.0),
    )

    # out-of-domain: recall doesn't apply
    if item.domain == "out_of_domain":
        row.evidence_recall_at_k = None

    return row