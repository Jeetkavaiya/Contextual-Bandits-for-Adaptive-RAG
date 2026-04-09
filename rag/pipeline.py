from __future__ import annotations

import time
from typing import Dict, List, Tuple

from .config import RAGConfig
from .hybrid_rank import hybrid_rank
from .index_bm25 import BM25Index
from .index_dense import DenseIndex
from .llm import OllamaLLM
from .prompt import build_answer_prompt, format_chat_history
from .rerank import CrossEncoderReranker
from .rewrite import OllamaRewriter
from .types import Chunk, Hit, RAGResult, TimingInfo, TokenUsage


def _best_by_chunk_id_keep_max_score(hits: List[Hit]) -> List[Hit]:
    best: Dict[str, Hit] = {}
    for h in hits:
        prev = best.get(h.chunk_id)
        if prev is None or h.score > prev.score:
            best[h.chunk_id] = h
    return list(best.values())


def _rough_token_estimate(text: str) -> int:
    return 0 if not text.strip() else max(1, len(text.split()))


class RAGPipeline:
    """
    Full baseline pipeline.

    LOCKED RULE:
    - refined query is used only for retrieval
    - final answer prompt uses: memory + raw_query + top-k chunks
    """

    def __init__(
        self,
        bm25: BM25Index,
        dense: DenseIndex,
        rewriter: OllamaRewriter,
        reranker: CrossEncoderReranker,
        llm: OllamaLLM,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rewriter = rewriter
        self.reranker = reranker
        self.llm = llm

    def run_rag(
        self,
        session_id: str,
        raw_query: str,
        chat_history: List[Tuple[str, str]],
        config: RAGConfig,
    ) -> RAGResult:
        t_all0 = time.perf_counter()
        timings = TimingInfo()
        tokens = TokenUsage()

        # Memory for final answer prompt
        memory_text = format_chat_history(chat_history, max_chars=config.memory_max_chars)

        # (1) Retrieval-only refine
        t0 = time.perf_counter()
        if config.rewrite:
            retrieval_query = self.rewriter.refine_query(
                raw_query=raw_query,
                chat_history=chat_history,
                memory_max_chars=config.memory_max_chars,
            )
        else:
            retrieval_query = raw_query.strip()
        timings.rewrite_s = time.perf_counter() - t0

        # (2) Multi-query list (retrieval-only)
        if config.rewrite:
            retrieval_queries = self.rewriter.multi_query(retrieval_query, nq=config.nq)
        else:
            retrieval_queries = [retrieval_query]

        # (3) Retrieve BM25 + Dense for each retrieval query
        t1 = time.perf_counter()
        all_bm25: List[Hit] = []
        all_dense: List[Hit] = []

        for q in retrieval_queries:
            all_bm25.extend(self.bm25.search(q, k=config.bm25_kcand))
            all_dense.extend(self.dense.search(q, k=config.dense_kcand))

        all_bm25 = _best_by_chunk_id_keep_max_score(all_bm25)
        all_dense = _best_by_chunk_id_keep_max_score(all_dense)
        timings.retrieve_s = time.perf_counter() - t1

        # (4) Hybrid combine (x*bm25_norm + y*dense_norm)
        t2 = time.perf_counter()
        pool_k = max(config.rerank_topR, config.k_final) if config.rerank else config.k_final
        hybrid_hits = hybrid_rank(
            bm25_hits=all_bm25,
            dense_hits=all_dense,
            x_bm25=config.x_bm25,
            y_dense=config.y_dense,
            topk=max(pool_k, 50),
        )
        timings.hybrid_s = time.perf_counter() - t2

        # (5) Optional rerank (use retrieval_query here)
        t3 = time.perf_counter()
        if config.rerank:
            reranked_hits = self.reranker.rerank(retrieval_query, hybrid_hits, topR=config.rerank_topR)
            context_hits = reranked_hits[: config.k_final]
        else:
            context_hits = hybrid_hits[: config.k_final]
        timings.rerank_s = time.perf_counter() - t3

        # (6) Final context chunks
        context_chunks: List[Chunk] = [
            Chunk(chunk_id=h.chunk_id, text=h.text, meta=h.meta) for h in context_hits
        ]

        # (7) Build final prompt (raw_query only)
        t4 = time.perf_counter()
        prompt = build_answer_prompt(raw_query=raw_query, memory_text=memory_text, context_chunks=context_chunks)
        timings.build_prompt_s = time.perf_counter() - t4

        # Cost estimates (optional; true counts come from Ollama prompt_eval_count)
        tokens.memory_tokens = _rough_token_estimate(memory_text)
        tokens.context_tokens = _rough_token_estimate("\n".join([c.text for c in context_chunks]))

        # (8) Generate answer
        answer, usage, gen_s = self.llm.generate(prompt)
        timings.gen_s = gen_s

        tokens.prompt_tokens = usage.prompt_tokens
        tokens.completion_tokens = usage.completion_tokens
        tokens.total_tokens = usage.total_tokens

        timings.total_s = time.perf_counter() - t_all0

        return RAGResult(
            raw_query=raw_query,
            retrieval_query=retrieval_query,
            retrieval_queries=retrieval_queries,
            answer=answer,
            context_chunks=context_chunks,
            context_hits=context_hits,
            timings=timings,
            tokens=tokens,
            meta={
                "session_id": session_id,
                "config": config.to_dict(),
                "num_bm25_candidates": len(all_bm25),
                "num_dense_candidates": len(all_dense),
                "num_hybrid_candidates": len(hybrid_hits),
            },
        )