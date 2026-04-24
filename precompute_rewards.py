"""
Phase 1 — Pre-compute the reward table for all (eval_item, action) pairs.

Why two phases?
  The full RAG pipeline + Ollama judge takes ~15-60 s per call.
  With 100 items × 60 actions = 6 000 calls we need parallelism.
  This script handles a *slice* of items (--start / --end), so you can
  run it as a SLURM array job (one task per item) and merge afterwards.

Output (per task)
  results/rewards/rewards_{start:04d}_{end:04d}.npz
    states  : float32 (n_items, 11)
    rewards : float32 (n_items, n_actions)
    item_ids: str     (n_items,)

After all tasks finish, run:
  python precompute_rewards.py --merge --out results/reward_table.npz

Usage examples
  # All items at once (local / single node)
  python precompute_rewards.py --evalset data/evalset_25_gold.jsonl

  # Item slice (SLURM array task i covers items [i*chunk, (i+1)*chunk))
  python precompute_rewards.py --evalset data/evalset_100_gold.jsonl \
      --start 0 --end 10

  # Merge partials into one table
  python precompute_rewards.py --merge --out results/reward_table.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Repository root (works whether script is run from repo root or slurm dir)
# ---------------------------------------------------------------------------
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
from rag.metrics import score_one

from env.action_space import build_action_space, action_to_config, Action
from env.state_features import extract_state, state_to_vector, STATE_KEYS


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def build_pipeline(
    pdf_path: str,
    cache_dir: str,
    ollama_host: str = "http://localhost:11434",
    embed_model: str = "nomic-embed-text",
    llm_model: Optional[str] = None,
) -> RAGPipeline:
    print(f"[setup] Loading chunks from {pdf_path} …", flush=True)
    chunks, _ = load_or_build_chunks(
        pdf_path=pdf_path,
        chunk_size=450,
        overlap=80,
        cache_dir=cache_dir,
    )
    print(f"[setup] {len(chunks)} chunks loaded.", flush=True)

    bm25 = BM25Index()
    bm25.build(chunks)
    print("[setup] BM25 index built.", flush=True)

    dense = DenseIndex(embed_model=embed_model, base_url=ollama_host)
    dense.build(chunks, batch_size=32, cache_dir=cache_dir, use_cache=True)
    print("[setup] Dense index built.", flush=True)

    rewriter = OllamaRewriter(base_url=ollama_host)
    reranker = CrossEncoderReranker()
    llm = OllamaLLM(base_url=ollama_host, model=llm_model)

    pipe = RAGPipeline(bm25=bm25, dense=dense, rewriter=rewriter,
                       reranker=reranker, llm=llm)
    print("[setup] Pipeline ready.", flush=True)
    return pipe


# ---------------------------------------------------------------------------
# Reward computation for one item × all actions
# ---------------------------------------------------------------------------

def compute_rewards_for_item(
    item: EvalItem,
    pipe: RAGPipeline,
    actions: List[Action],
    base_cfg: RAGConfig,
    judge: OllamaJudge,
    reward_weights: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    state_vec : (11,)  float32
    reward_vec: (K,)   float32   K = len(actions)
    """
    # State is cheap — compute once
    st = extract_state(
        raw_query=item.raw_query,
        bm25=pipe.bm25,
        dense=pipe.dense,
    )
    state_vec = np.array(state_to_vector(st), dtype=np.float32)

    reward_vec = np.full(len(actions), np.nan, dtype=np.float32)
    w = reward_weights

    for a_idx, action in enumerate(actions):
        cfg = action_to_config(action, base=base_cfg)
        try:
            res = pipe.run_rag(
                session_id=item.qid,
                raw_query=item.raw_query,
                chat_history=item.chat_history,
                config=cfg,
            )
            row = score_one(item, res, judge=judge)

            correctness  = (float(row.correctness) / 2.0)   if row.correctness  is not None else 0.0
            faithfulness = float(row.faithfulness)           if row.faithfulness is not None else 0.0
            recall       = float(row.evidence_recall_at_k)  if row.evidence_recall_at_k is not None else 0.0

            t       = float(row.total_time_s)
            tok_k   = (float(row.total_tokens) / 1000.0) if row.total_tokens is not None else 0.0
            k_final = float(row.k_final)

            reward = (
                w["w_correctness"] * correctness
                + w["w_faithfulness"] * faithfulness
                + w["w_recall"] * recall
                - w["w_time"] * t
                - w["w_tokens"] * tok_k
                - w["w_chunks"] * k_final
                - w["w_rerank"] * float(row.rerank_enabled)
                - w["w_rewrite"] * float(row.rewrite_enabled)
            )
            reward_vec[a_idx] = float(reward)
        except Exception as exc:
            print(f"  [WARN] item={item.qid} action={a_idx} failed: {exc}", flush=True)

    return state_vec, reward_vec


# ---------------------------------------------------------------------------
# Main: compute slice
# ---------------------------------------------------------------------------

def run_precompute(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items_all = load_evalset_jsonl(args.evalset)
    items = items_all[args.start : args.end]
    print(f"[precompute] Processing items [{args.start}:{args.end}] "
          f"({len(items)} items)", flush=True)

    if len(items) == 0:
        print("[precompute] No items in slice. Exiting.", flush=True)
        return

    pipe = build_pipeline(
        pdf_path=str(REPO_ROOT / "data" / "prudentservices_dataset_rag.pdf"),
        cache_dir=str(REPO_ROOT / "data" / "cache"),
        ollama_host=args.ollama_host,
        embed_model=args.embed_model,
        llm_model=args.llm_model,
    )
    os.environ["OLLAMA_BASE_URL"] = args.ollama_host
    judge = OllamaJudge()
    actions = build_action_space()
    base_cfg = RAGConfig()

    reward_weights = {
        "w_correctness": 1.0,
        "w_faithfulness": 0.5,
        "w_recall": 0.5,
        "w_time": 0.05,
        "w_tokens": 0.002,
        "w_chunks": 0.01,
        "w_rerank": 0.05,
        "w_rewrite": 0.02,
    }

    n_items   = len(items)
    n_actions = len(actions)
    states  = np.zeros((n_items, len(STATE_KEYS)), dtype=np.float32)
    rewards = np.full((n_items, n_actions), np.nan, dtype=np.float32)
    ids     = np.array([it.qid for it in items])

    for i, item in enumerate(items):
        t0 = time.time()
        print(f"[{i+1}/{n_items}] item={item.qid} …", flush=True)
        s, r = compute_rewards_for_item(
            item, pipe, actions, base_cfg, judge, reward_weights
        )
        states[i]  = s
        rewards[i] = r
        print(f"  done in {time.time()-t0:.1f}s  "
              f"valid={np.sum(~np.isnan(r))}/{n_actions}", flush=True)

    out_file = out_dir / f"rewards_{args.start:04d}_{args.end:04d}.npz"
    np.savez_compressed(str(out_file), states=states, rewards=rewards, ids=ids)
    print(f"[precompute] Saved → {out_file}", flush=True)

    # Also save action metadata once
    meta_file = out_dir / "actions_meta.json"
    if not meta_file.exists():
        import dataclasses
        meta = [dataclasses.asdict(a) for a in actions]
        meta_file.write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Merge partials
# ---------------------------------------------------------------------------

def run_merge(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    files = sorted(out_dir.glob("rewards_*.npz"))
    if not files:
        print(f"[merge] No reward files found in {out_dir}"); return

    all_states, all_rewards, all_ids = [], [], []
    for fpath in files:
        data = np.load(str(fpath), allow_pickle=True)
        all_states.append(data["states"])
        all_rewards.append(data["rewards"])
        all_ids.append(data["ids"])
        print(f"  loaded {fpath.name}  shape={data['rewards'].shape}", flush=True)

    states  = np.concatenate(all_states,  axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    ids     = np.concatenate(all_ids,     axis=0)

    out_file = Path(args.merge_out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_file), states=states, rewards=rewards, ids=ids)
    print(f"[merge] Saved combined table → {out_file}  "
          f"shape={rewards.shape}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute RAG reward table")
    sub = parser.add_subparsers(dest="cmd")

    # --- compute sub-command ------------------------------------------------
    p_comp = sub.add_parser("compute", help="Run RAG pipeline for item slice")
    p_comp.add_argument("--evalset",    default="data/evalset_25_gold.jsonl")
    p_comp.add_argument("--start",      type=int, default=0,
                        help="First item index (inclusive)")
    p_comp.add_argument("--end",        type=int, default=None,
                        help="Last item index (exclusive). Default: all items")
    p_comp.add_argument("--out_dir",    default="results/rewards",
                        help="Directory for partial .npz files")
    p_comp.add_argument("--ollama_host", default="http://localhost:11434")
    p_comp.add_argument("--embed_model", default="nomic-embed-text")
    p_comp.add_argument("--llm_model",   default=None,
                        help="Ollama model name. None = pipeline default.")

    # --- merge sub-command --------------------------------------------------
    p_merge = sub.add_parser("merge", help="Merge partial reward .npz files")
    p_merge.add_argument("--out_dir",   default="results/rewards")
    p_merge.add_argument("--merge_out", default="results/reward_table.npz")

    args = parser.parse_args()

    if args.cmd == "compute":
        # Resolve default end index
        if args.end is None:
            items_tmp = load_evalset_jsonl(args.evalset)
            args.end = len(items_tmp)
        run_precompute(args)
    elif args.cmd == "merge":
        run_merge(args)
    else:
        # Backward-compat: no subcommand → run compute with all items
        class _Args:
            cmd = "compute"
            evalset    = "data/evalset_25_gold.jsonl"
            start      = 0
            end        = None
            out_dir    = "results/rewards"
            ollama_host= "http://localhost:11434"
            embed_model= "nomic-embed-text"
            llm_model  = None
        a = _Args()
        items_tmp = load_evalset_jsonl(a.evalset)
        a.end = len(items_tmp)
        run_precompute(a)
