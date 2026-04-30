# Structure
_Last updated: 2026-04-30_

## Summary
Flat Python project organized into three top-level packages (`bandits/`, `env/`, `rag/`) plus root-level scripts for each pipeline phase. No package installation — scripts manipulate `sys.path` directly. SLURM job scripts live in `slurm/`, data and results in `data/` and `results/`.

## Directory Layout

```
final-project-team_4/
│
├── precompute_rewards.py   # Phase 1: exhaustive (query × action) reward computation
├── train_bandit.py         # Phase 2: bandit training on precomputed reward table
├── analyze_results.py      # Phase 3: plotting and analysis
├── requirements_ccr.txt    # Pinned deps for UB CCR HPC cluster
├── README.md               # Project overview + CCR setup guide
│
├── bandits/                # Bandit algorithm implementations
│   ├── __init__.py         # Re-exports: EpsilonGreedy, UCB1, LinUCBDisjoint, LinearThompsonSampling
│   ├── base.py             # BaseBandit abstract class
│   ├── epsilon_greedy.py   # ε-greedy with optional decay
│   ├── ucb.py              # UCB1
│   ├── linucb.py           # LinUCB Disjoint (contextual)
│   └── thompson.py         # Linear Thompson Sampling (contextual)
│
├── env/                    # RL environment
│   ├── action_space.py     # Action dataclass + build_action_space() → 60 actions
│   ├── rag_env.py          # RAGEnv: 1-step MDP wrapping RAGPipeline
│   └── state_features.py   # extract_state() → 11-dim vector, STATE_KEYS
│
├── rag/                    # RAG pipeline components
│   ├── __init__.py
│   ├── config.py           # RAGConfig dataclass (all pipeline knobs)
│   ├── types.py            # Shared types: Chunk, RAGResult, Timings, Tokens
│   ├── pipeline.py         # RAGPipeline: main orchestrator
│   ├── pdf_ingest.py       # PDF → chunks (with caching)
│   ├── index_bm25.py       # BM25Index (rank-bm25)
│   ├── index_dense.py      # DenseIndex (FAISS + Ollama embeddings)
│   ├── hybrid_rank.py      # Hybrid BM25+dense score fusion
│   ├── rerank.py           # CrossEncoderReranker (BAAI/bge-reranker-base)
│   ├── rewrite.py          # OllamaRewriter: query expansion
│   ├── llm.py              # OllamaLLM: generation via local Ollama
│   ├── prompt.py           # Prompt templates
│   ├── evalset.py          # EvalItem dataclass + load_evalset_jsonl()
│   ├── evaluate.py         # Batch evaluation helpers
│   ├── judge.py            # OllamaJudge: LLM-as-judge scoring
│   ├── metrics.py          # MetricsRow + score_one()
│   └── reporting.py        # Results reporting/formatting
│
├── slurm/                  # HPC job scripts for UB CCR cluster
│   ├── 0_precompute.sh     # SLURM array job: precompute rewards in parallel
│   ├── 1_merge.sh          # Merge reward shards
│   └── 2_train_bandit.sh   # CPU training job
│
├── data/                   # Input data (gitignored or local)
│   ├── evalset_25_gold.jsonl   # Small eval set (25 items)
│   └── evalset_100_gold.jsonl  # Full eval set (100 items)
│   └── *.pdf                   # Source documents for RAG
│
├── results/                # Output artifacts
│   ├── rewards/            # Partial reward shards from SLURM array tasks
│   │   └── rewards_NNNN_MMMM.npz
│   ├── reward_table.npz    # Merged reward table (N_items × 60_actions)
│   ├── bandit_results.npz  # Per-step training arrays for all 4 algorithms
│   └── bandit_summary.json # Final metrics table
│
├── checkpoints/            # Saved bandit agent states (.pkl per algorithm)
│
└── notebooks/              # Jupyter notebooks (exploratory analysis)
```

## Key Files by Role

| File | Role |
|------|------|
| `train_bandit.py` | Main training entry point; loads reward table, runs 4 agents, saves results |
| `precompute_rewards.py` | Reward table generator; supports slice mode for SLURM parallelism |
| `analyze_results.py` | Loads bandit_results.npz, produces comparison plots |
| `env/rag_env.py` | 1-step RL environment (used during precompute, not training) |
| `env/action_space.py` | Defines the 60-action discrete space |
| `env/state_features.py` | 11-dim state vector definition (STATE_KEYS is the canonical ordering) |
| `rag/pipeline.py` | Core RAG orchestrator — the expensive component precompute avoids |
| `rag/config.py` | RAGConfig — single source of truth for all pipeline knobs |
| `rag/judge.py` | LLM-as-judge (Ollama) for correctness/faithfulness scoring |
| `rag/metrics.py` | `score_one()` combines judge output + timing into scalar reward |
| `bandits/base.py` | `BaseBandit` ABC — all algorithms implement `select()`, `update()`, `save()`, `load()` |

## Where to Add New Things

| Task | Location |
|------|----------|
| New bandit algorithm | `bandits/<name>.py` + export from `bandits/__init__.py` |
| New state feature | `env/state_features.py` — add to `STATE_KEYS` and `extract_state()` |
| New action dimension | `env/action_space.py` — extend `Action` dataclass and `build_action_space()` |
| New RAG component | `rag/<component>.py` + wire into `rag/pipeline.py` |
| New eval metric | `rag/metrics.py` + `MetricsRow` + `score_one()` |
| New reward weight | `env/rag_env.py` `self.w` dict |

## Configuration Files

| File | Purpose |
|------|---------|
| `requirements_ccr.txt` | Pinned Python deps for CCR HPC (no lockfile for local dev) |
| `slurm/0_precompute.sh` | SLURM directives + GPU detection logic for precompute jobs |
| `slurm/1_merge.sh` | SLURM job for merging reward shards |
| `slurm/2_train_bandit.sh` | SLURM job for CPU bandit training |

## Gaps & Unknowns
- No `setup.py`, `pyproject.toml`, or `setup.cfg` — not installable as a package.
- `notebooks/` contents not explored.
- `data/` and `results/` likely gitignored (large files).
