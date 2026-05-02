"""
Apples-to-apples comparison runner.

For each query:
  1) Run the FIXED baseline RAG (RAGConfig defaults).
  2) Run the trained RL policy (default: Thompson Sampling) — pick action via
     the loaded checkpoint and run RAG with that config.

Both runs hit the same Ollama, the same indices, the same judge, so the only
difference is the chosen config per query.

Outputs:
  results/compare/baseline_metrics.csv
  results/compare/rl_metrics.csv
  results/compare/details.jsonl
  results/compare/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from rag.pdf_ingest import load_or_build_chunks
from rag.index_bm25 import BM25Index
from rag.index_dense import DenseIndex
from rag.rewrite import OllamaRewriter
from rag.rerank import CrossEncoderReranker
from rag.llm import OllamaLLM
from rag.pipeline import RAGPipeline
from rag.config import RAGConfig
from rag.evalset import load_evalset_jsonl, EvalItem
from rag.judge import OllamaJudge
from rag.metrics import score_one, MetricsRow

from env.action_space import build_action_space, action_to_config
from env.state_features import extract_state, state_to_vector, STATE_KEYS
from bandits import EpsilonGreedy, UCB1, LinUCBDisjoint, LinearThompsonSampling


# ---------------------------------------------------------------------------
# Selected 10 queries spanning different domains / types
# ---------------------------------------------------------------------------
SELECTED_QIDS: List[str] = [
    # already in baseline_5 — RL-only here
    "in_010",   # in_domain, list, no chat
    "in_065",   # in_domain, memory-dependent w/ chat
    "in_062",   # in_domain, procedural w/ chat
    "ood_001",  # OOD, factual
    "ood_002",  # OOD, math
    # NEW — both baseline and RL
    "in_002",   # in_domain, single-fact lookup
    "in_046",   # in_domain, structured list
    "in_006",   # in_domain, single-fact lookup
    "ood_005",  # OOD, factual (boiling point)
    "ood_003",  # OOD, math (15% of 260)
]


# ---------------------------------------------------------------------------
# Build pipeline (shared for both methods)
# ---------------------------------------------------------------------------
def build_pipeline(
    pdf_path: str,
    cache_dir: str,
    ollama_host: str = "http://127.0.0.1:11434",
    embed_model: str = "nomic-embed-text",
    llm_model: str | None = None,
) -> RAGPipeline:
    print(f"[setup] Loading chunks from {pdf_path} …", flush=True)
    chunks, _ = load_or_build_chunks(
        pdf_path=pdf_path, chunk_size=450, overlap=80, cache_dir=cache_dir,
    )
    print(f"[setup] {len(chunks)} chunks loaded.", flush=True)

    bm25 = BM25Index(); bm25.build(chunks)
    dense = DenseIndex(embed_model=embed_model, base_url=ollama_host)
    dense.build(chunks, batch_size=32, cache_dir=cache_dir, use_cache=True)
    print("[setup] Indices built.", flush=True)

    rewriter = OllamaRewriter(base_url=ollama_host)
    reranker = CrossEncoderReranker()
    llm = OllamaLLM(base_url=ollama_host, model=llm_model)
    pipe = RAGPipeline(bm25=bm25, dense=dense, rewriter=rewriter,
                       reranker=reranker, llm=llm)
    print("[setup] Pipeline ready.", flush=True)
    return pipe


# ---------------------------------------------------------------------------
# Load trained bandit checkpoint
# ---------------------------------------------------------------------------
def load_bandit(name: str, ckpt_dir: str, K: int, d: int):
    """Recreate an empty agent of the right type, then load state from .pkl."""
    if name == "thompson":
        agent = LinearThompsonSampling(K, d=d)
    elif name == "linucb":
        agent = LinUCBDisjoint(K, d=d)
    elif name == "epsilon_greedy":
        agent = EpsilonGreedy(K)
    elif name == "ucb1":
        agent = UCB1(K)
    else:
        raise ValueError(f"Unknown bandit: {name}")
    agent.load(str(Path(ckpt_dir) / f"{name}.pkl"))
    return agent


# ---------------------------------------------------------------------------
# State scaler from saved bandit_results
# ---------------------------------------------------------------------------
class FixedScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(float)
        self.std = std.astype(float)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.std


# ---------------------------------------------------------------------------
# Reward (must match env/rag_env.py)
# ---------------------------------------------------------------------------
REWARD_W = {
    "w_correctness": 1.0, "w_faithfulness": 0.5, "w_recall": 0.5,
    "w_time": 0.05, "w_tokens": 0.002, "w_chunks": 0.01,
    "w_rerank": 0.05, "w_rewrite": 0.02,
}


def reward_from_row(row: MetricsRow) -> float:
    correctness = (float(row.correctness) / 2.0) if row.correctness is not None else 0.0
    faithfulness = float(row.faithfulness) if row.faithfulness is not None else 0.0
    recall = float(row.evidence_recall_at_k) if row.evidence_recall_at_k is not None else 0.0
    t = float(row.total_time_s)
    tok_k = (float(row.total_tokens) / 1000.0) if row.total_tokens is not None else 0.0
    k_final = float(row.k_final)
    w = REWARD_W
    return (
        w["w_correctness"] * correctness
        + w["w_faithfulness"] * faithfulness
        + w["w_recall"] * recall
        - w["w_time"] * t
        - w["w_tokens"] * tok_k
        - w["w_chunks"] * k_final
        - w["w_rerank"] * float(row.rerank_enabled)
        - w["w_rewrite"] * float(row.rewrite_enabled)
    )


# ---------------------------------------------------------------------------
# Run one (item, config) pair through the pipeline + judge
# ---------------------------------------------------------------------------
def run_one(item: EvalItem, cfg: RAGConfig, pipe: RAGPipeline, judge: OllamaJudge):
    res = pipe.run_rag(
        session_id=item.qid,
        raw_query=item.raw_query,
        chat_history=item.chat_history,
        config=cfg,
    )
    row = score_one(item, res, judge=judge)
    reward = reward_from_row(row)
    return row, res, reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evalset", default="data/evalset_25_gold.jsonl")
    p.add_argument("--bandit", default="thompson",
                   choices=["thompson", "linucb", "epsilon_greedy", "ucb1"])
    p.add_argument("--ckpt_dir", default="checkpoints/task_0")
    p.add_argument("--bandit_results", default="results/task_0/bandit_results.npz",
                   help="Used only to recover state mean/std for scaling.")
    p.add_argument("--out_dir", default="results/compare")
    p.add_argument("--ollama_host", default="http://127.0.0.1:11434")
    p.add_argument("--qids", nargs="*", default=None,
                   help="Override SELECTED_QIDS")
    p.add_argument("--skip_baseline_qids", nargs="*", default=None,
                   help="Skip baseline run for these qids (use existing numbers)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only first N qids (smoke test)")
    p.add_argument("--rng_seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.rng_seed)

    qids = args.qids if args.qids else SELECTED_QIDS
    if args.limit is not None:
        qids = qids[: args.limit]
    skip_base = set(args.skip_baseline_qids or [])

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compare] qids = {qids}", flush=True)
    print(f"[compare] skip baseline for = {sorted(skip_base)}", flush=True)
    print(f"[compare] bandit            = {args.bandit}", flush=True)

    # 1. Load eval set
    items_all = load_evalset_jsonl(args.evalset)
    by_qid = {it.qid: it for it in items_all}
    missing = [q for q in qids if q not in by_qid]
    if missing:
        sys.exit(f"[ERROR] Missing qids in evalset: {missing}")
    items = [by_qid[q] for q in qids]

    # 2. Build pipeline + judge
    os.environ["OLLAMA_BASE_URL"] = args.ollama_host
    pipe = build_pipeline(
        pdf_path=str(REPO_ROOT / "data" / "prudentservices_dataset_rag.pdf"),
        cache_dir=str(REPO_ROOT / "data" / "cache"),
        ollama_host=args.ollama_host,
    )
    judge = OllamaJudge()

    # 3. Load bandit policy + state scaler
    actions = build_action_space()
    K = len(actions); d = len(STATE_KEYS)
    agent = load_bandit(args.bandit, args.ckpt_dir, K, d)
    print(f"[compare] loaded {args.bandit} (K={K}, d={d}, total_pulls={agent.t})",
          flush=True)

    bres = np.load(str(REPO_ROOT / args.bandit_results), allow_pickle=True)
    scaler = FixedScaler(bres["state_mean"], bres["state_std"])

    # 4. Run baseline + RL for each query
    base_cfg = RAGConfig()  # fixed baseline
    rows_base: List[Dict[str, Any]] = []
    rows_rl:   List[Dict[str, Any]] = []
    details_lines: List[str] = []

    # Reuse existing baseline_5 numbers when available + skip is requested
    existing_baseline_path = REPO_ROOT / "results" / "baseline_5_details.jsonl"
    existing_baseline: Dict[str, dict] = {}
    if existing_baseline_path.exists():
        for ln in existing_baseline_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                obj = json.loads(ln)
                existing_baseline[obj["qid"]] = obj

    for idx, item in enumerate(items):
        print(f"\n=== [{idx+1}/{len(items)}] {item.qid} ({item.domain}) ===", flush=True)
        print(f"    raw_query: {item.raw_query!r}", flush=True)

        # ---------- BASELINE ----------
        if item.qid in skip_base and item.qid in existing_baseline:
            print(f"  [BASELINE] reusing existing run from baseline_5_details.jsonl",
                  flush=True)
            mr = existing_baseline[item.qid]["metrics_row"]
            row_base = MetricsRow(**mr)
            res_base_dict = existing_baseline[item.qid]
            rwd_base = reward_from_row(row_base)
        else:
            t0 = time.time()
            print(f"  [BASELINE] running fixed config: rewrite=1 nq=3 rerank=1 k=10 x_bm25=0.5",
                  flush=True)
            row_base, res_base, rwd_base = run_one(item, base_cfg, pipe, judge)
            print(f"  [BASELINE] done in {time.time()-t0:.1f}s  "
                  f"correctness={row_base.correctness} faithfulness={row_base.faithfulness} "
                  f"recall={row_base.evidence_recall_at_k} latency={row_base.total_time_s:.1f}s "
                  f"tokens={row_base.total_tokens} reward={rwd_base:.3f}",
                  flush=True)
            res_base_dict = {
                "answer": res_base.answer,
                "context_chunk_ids": [c.chunk_id for c in res_base.context_chunks],
                "retrieval_query": res_base.retrieval_query,
            }

        # ---------- RL ----------
        # Extract state, scale, pick action
        st = extract_state(raw_query=item.raw_query, bm25=pipe.bm25, dense=pipe.dense)
        s_raw = np.array(state_to_vector(st), dtype=float)
        s_norm = scaler.transform(s_raw)

        # contextual bandits get the scaled state; non-contextual ignore it
        if args.bandit in ("thompson", "linucb"):
            a_idx = int(agent.select(s_norm))
        else:
            a_idx = int(agent.select(None))
        a = actions[a_idx]
        rl_cfg = action_to_config(a, base=base_cfg)

        t1 = time.time()
        print(f"  [RL/{args.bandit}] state(raw)={[round(v,2) for v in s_raw]}", flush=True)
        print(f"  [RL/{args.bandit}] picked arm {a_idx}: rewrite={a.rewrite} nq={a.nq} "
              f"rerank={a.rerank} k={a.k_final} x_bm25={a.x_bm25}", flush=True)
        row_rl, res_rl, rwd_rl = run_one(item, rl_cfg, pipe, judge)
        print(f"  [RL/{args.bandit}] done in {time.time()-t1:.1f}s  "
              f"correctness={row_rl.correctness} faithfulness={row_rl.faithfulness} "
              f"recall={row_rl.evidence_recall_at_k} latency={row_rl.total_time_s:.1f}s "
              f"tokens={row_rl.total_tokens} reward={rwd_rl:.3f}", flush=True)

        # Collect rows
        rows_base.append({
            "qid": item.qid, "domain": item.domain, "method": "baseline",
            **{k: getattr(row_base, k) for k in (
                "correctness","faithfulness","evidence_recall_at_k",
                "total_time_s","total_tokens","prompt_tokens","completion_tokens",
                "k_final","nq","rerank_enabled","rewrite_enabled","x_bm25","y_dense",
            )},
            "reward": rwd_base,
        })
        rows_rl.append({
            "qid": item.qid, "domain": item.domain, "method": f"rl_{args.bandit}",
            "arm_idx": a_idx,
            **{k: getattr(row_rl, k) for k in (
                "correctness","faithfulness","evidence_recall_at_k",
                "total_time_s","total_tokens","prompt_tokens","completion_tokens",
                "k_final","nq","rerank_enabled","rewrite_enabled","x_bm25","y_dense",
            )},
            "reward": rwd_rl,
        })
        details_lines.append(json.dumps({
            "qid": item.qid, "domain": item.domain,
            "raw_query": item.raw_query,
            "gold_answer": item.gold_answer,
            "state_raw": s_raw.tolist(),
            "state_norm": s_norm.tolist(),
            "baseline": {
                "config": base_cfg.to_dict(),
                "answer": res_base_dict.get("answer"),
                "metrics_row": asdict(row_base),
                "reward": rwd_base,
            },
            "rl": {
                "bandit": args.bandit,
                "arm_idx": a_idx,
                "config": rl_cfg.to_dict(),
                "answer": res_rl.answer,
                "metrics_row": asdict(row_rl),
                "reward": rwd_rl,
            },
        }))

    # 5. Save artifacts
    import csv
    base_csv = out_dir / "baseline_metrics.csv"
    rl_csv = out_dir / f"rl_{args.bandit}_metrics.csv"
    details_path = out_dir / f"details_{args.bandit}.jsonl"

    def write_csv(path: Path, rows: List[Dict]) -> None:
        if not rows: return
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    write_csv(base_csv, rows_base)
    write_csv(rl_csv, rows_rl)
    details_path.write_text("\n".join(details_lines) + "\n", encoding="utf-8")

    # 6. Compute summary
    def aggr(rows: List[Dict], keys: List[str]) -> Dict[str, float]:
        out = {}
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            out[k] = float(np.mean(vals)) if vals else None  # type: ignore
        return out

    in_dom_idx = [i for i,r in enumerate(rows_base) if r["domain"] == "in_domain"]
    keys = ["correctness","faithfulness","evidence_recall_at_k",
            "total_time_s","total_tokens","k_final","reward"]
    summary = {
        "n_queries": len(items),
        "qids": qids,
        "bandit": args.bandit,
        "baseline_all": aggr(rows_base, keys),
        "rl_all":       aggr(rows_rl, keys),
        "baseline_in_domain": aggr([rows_base[i] for i in in_dom_idx], keys),
        "rl_in_domain":       aggr([rows_rl[i] for i in in_dom_idx], keys),
    }
    (out_dir / f"summary_{args.bandit}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 78, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
