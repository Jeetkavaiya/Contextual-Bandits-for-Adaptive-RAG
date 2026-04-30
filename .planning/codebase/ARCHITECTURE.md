# Architecture
_Last updated: 2026-04-30_

## Summary
This project frames RAG (Retrieval-Augmented Generation) configuration selection as a contextual bandit problem. A reward table is precomputed offline by exhaustively running all (query, action) pairs through the full RAG+LLM pipeline, then four bandit algorithms are trained against this table without needing to re-run Ollama. The split into a GPU-heavy precompute phase and a lightweight CPU training phase is driven by HPC (SLURM) parallelism requirements.

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│  Phase 0: Data Ingestion                                 │
│  PDF → chunks → BM25 index + FAISS dense index          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Phase 1: Precompute Rewards  (precompute_rewards.py)    │
│  For each eval item × each action (60 configs):          │
│    1. extract_state(query) → 11-dim state vector         │
│    2. RAGPipeline.run_rag(query, config) → answer        │
│    3. OllamaJudge.score(answer) → correctness/faithfulness│
│    4. score_one() → MetricsRow → scalar reward           │
│  Output: rewards_{start}_{end}.npz per SLURM task        │
└──────────────────────┬──────────────────────────────────┘
                       │ merge step
┌──────────────────────▼──────────────────────────────────┐
│  Merged Reward Table: results/reward_table.npz           │
│  Shape: (N_queries, 60_actions) rewards + (N, 11) states │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Phase 2: Bandit Training  (train_bandit.py)             │
│  4 algorithms trained on reward table:                   │
│  - EpsilonGreedy  (non-contextual baseline)              │
│  - UCB1           (non-contextual baseline)              │
│  - LinUCBDisjoint (contextual — main algorithm)          │
│  - LinearThompsonSampling (contextual — Bayesian)        │
│  Output: bandit_results.npz, bandit_summary.json, .pkl   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  Phase 3: Analysis  (analyze_results.py)                 │
│  Plot cumulative regret curves, reward distributions     │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### Action Space (`env/action_space.py`)
- 60 discrete actions = combinatorial product of RAG config knobs:
  - `rewrite` ∈ {0, 1}
  - `nq` ∈ {1, 3} (forced nq=1 when rewrite=0)
  - `rerank` ∈ {0, 1}
  - `k_final` ∈ {5, 10, 20}
  - `x_bm25` ∈ {0.2, 0.5, 0.8} (y_dense = 1 - x_bm25)
- `action_to_config()` maps Action → RAGConfig for the pipeline

### State Features (`env/state_features.py`)
- 11-dimensional vector per query (STATE_KEYS):
  - Query lexical features: `len_words`, `len_chars`
  - Query type signals: `has_list`, `has_steps`, `has_compare`, `has_why`, `has_math`
  - Retrieval confidence probes (k=5): `bm25_top`, `bm25_gap`, `dense_top`, `dense_gap`
- State is extracted cheaply from the raw query (before rewriting)

### RL Environment (`env/rag_env.py`)
- 1-step episodic MDP: each episode = one query
- `reset(item)` → state vector; `step(action_idx)` → (next_state, reward, done=True, info)
- Reward formula: `correctness/2 + 0.5*faithfulness + 0.5*recall - cost_penalties`
- Used by `precompute_rewards.py` to generate the reward table; NOT used during training

### RAG Pipeline (`rag/pipeline.py`)
- Orchestrates: query rewrite → hybrid retrieval → rerank → LLM generation
- Components: `OllamaRewriter`, `BM25Index`, `DenseIndex`, `HybridRanker`, `CrossEncoderReranker`, `OllamaLLM`
- All RAG config knobs are set per-call via `RAGConfig`

### Bandit Algorithms (`bandits/`)
- `BaseBandit`: abstract base with `select(context)` / `update(arm, reward, context)` / `save()` / `load()`
- `EpsilonGreedy`: ε-greedy with optional annealing decay
- `UCB1`: upper confidence bound, non-contextual
- `LinUCBDisjoint`: one linear model per arm (disjoint variant), ridge-regularized
- `LinearThompsonSampling`: Bayesian linear regression, posterior sampling per arm

### Reward Function
```
reward = w_correctness * (correctness/2)
       + w_faithfulness * faithfulness
       + w_recall * evidence_recall_at_k
       - w_time * total_time_s
       - w_tokens * (total_tokens/1000)
       - w_chunks * k_final
       - w_rerank * rerank_enabled
       - w_rewrite * rewrite_enabled
```
Default weights: correctness=1.0, faithfulness=0.5, recall=0.5, time=0.05, tokens=0.002, chunks=0.01, rerank=0.05, rewrite=0.02

## Data Flow

```
data/evalset_*_gold.jsonl   →  EvalItem list
data/*.pdf                  →  chunks (via pdf_ingest)
chunks                      →  BM25Index + DenseIndex (FAISS)
(query, action)             →  RAGPipeline.run_rag() → RAGResult
RAGResult + EvalItem        →  score_one() → MetricsRow → reward scalar
(query, state_vec, reward)  →  reward_table.npz
reward_table.npz            →  bandit training → bandit_results.npz
bandit_results.npz          →  analyze_results.py → plots
```

## Execution Modes

| Mode | Script | Compute | Notes |
|------|--------|---------|-------|
| Single-node precompute | `precompute_rewards.py` | GPU (Ollama) | Slow, for local dev |
| SLURM array precompute | `slurm/0_precompute.sh` | GPU cluster | Array job, 1 task per slice |
| Merge partials | `precompute_rewards.py --merge` | CPU | Concatenates .npz shards |
| Bandit training | `train_bandit.py` | CPU | Pure NumPy, fast |
| Analysis | `analyze_results.py` | CPU | Matplotlib plots |

## Gaps & Unknowns
- `rag/pipeline.py` and `rag/evaluate.py` not read in detail — exact RAG orchestration flow partially inferred.
- Number of eval items in production run unclear (scripts show 25 and 100 as examples).
- `rag/reporting.py` purpose not fully explored.
