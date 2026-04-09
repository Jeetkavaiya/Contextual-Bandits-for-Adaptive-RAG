from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List, Optional

from .config import RAGConfig
from .evalset import EvalItem
from .judge import OllamaJudge
from .metrics import MetricsRow, score_one
from .pipeline import RAGPipeline


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def evaluate_items(
    pipe: RAGPipeline,
    config: RAGConfig,
    items: List[EvalItem],
    output_dir: str = "outputs",
    judge_model: Optional[str] = None,
    run_name: str = "baseline",
) -> None:
    """
    Runs pipe.run_rag(...) on each EvalItem, scores with Ollama judge, and saves:
      outputs/<run_name>_metrics.csv
      outputs/<run_name>_details.jsonl
    """
    _ensure_dir(output_dir)

    judge = OllamaJudge(model=judge_model)

    details_path = os.path.join(output_dir, f"{run_name}_details.jsonl")
    metrics_path = os.path.join(output_dir, f"{run_name}_metrics.csv")

    rows: List[MetricsRow] = []

    with open(details_path, "w", encoding="utf-8") as f_details:
        for it in items:
            res = pipe.run_rag(
                session_id=it.qid,
                raw_query=it.raw_query,
                chat_history=it.chat_history,
                config=config,
            )

            row = score_one(it, res, judge=judge)
            rows.append(row)

            detail_obj = {
                "qid": it.qid,
                "domain": it.domain,
                "raw_query": it.raw_query,
                "gold_answer": it.gold_answer,
                "gold_support_chunk_ids": it.gold_support_chunk_ids,
                "retrieval_query": res.retrieval_query,
                "retrieval_queries": res.retrieval_queries,
                "answer": res.answer,
                "context_chunk_ids": [c.chunk_id for c in res.context_chunks],
                "context_pages": [c.meta.get("page") for c in res.context_chunks],
                "timings": asdict(res.timings),
                "tokens": asdict(res.tokens),
                "config": res.meta.get("config", {}),
            }
            f_details.write(json.dumps(detail_obj, ensure_ascii=False))
            f_details.write("\n")

    # Save metrics CSV (no pandas required)
    _save_metrics_csv(rows, metrics_path)

    # Print summary
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {details_path}\n")
    print_summary(rows)


def _save_metrics_csv(rows: List[MetricsRow], path: str) -> None:
    import csv

    if not rows:
        return

    fieldnames = list(asdict(rows[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def print_summary(rows: List[MetricsRow]) -> None:
    if not rows:
        print("No rows to summarize.")
        return

    def _filter(domain: str) -> List[MetricsRow]:
        return [r for r in rows if r.domain == domain]

    def _summ(tag: str, rr: List[MetricsRow]) -> None:
        corr = [float(r.correctness) for r in rr if r.correctness is not None]
        faith = [float(r.faithfulness) for r in rr if r.faithfulness is not None]
        recall = [float(r.evidence_recall_at_k) for r in rr if r.evidence_recall_at_k is not None]
        t = [float(r.total_time_s) for r in rr]
        tok = [float(r.total_tokens) for r in rr if r.total_tokens is not None]

        print(f"--- {tag} ---")
        print(f"n = {len(rr)}")
        print(f"avg_correctness (0-2): {round(_mean(corr), 3)}")
        print(f"avg_faithfulness (0-1): {round(_mean(faith), 3)}")
        if recall:
            print(f"evidence_recall@k (0-1): {round(_mean(recall), 3)}")
        print(f"avg_total_time_s: {round(_mean(t), 3)}")
        if tok:
            print(f"avg_total_tokens: {round(_mean(tok), 1)}")
        print()

    _summ("ALL", rows)
    _summ("IN_DOMAIN", _filter("in_domain"))
    _summ("OUT_OF_DOMAIN", _filter("out_of_domain"))