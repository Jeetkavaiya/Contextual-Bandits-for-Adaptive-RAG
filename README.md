# RL-RAG: Contextual Bandit Optimization for RAG Pipelines

**CSE 546 — Reinforcement Learning | University at Buffalo | Spring 2026**

**Team 4:** Jeet Kavaiya · Dev Desai · Vansh Thakkar

---

## Overview

This project applies **contextual multi-armed bandit algorithms** to optimize Retrieval-Augmented Generation (RAG) pipelines. Instead of using a fixed RAG configuration for every query, a bandit agent learns to select the best pipeline configuration (chunk count, hybrid retrieval weights, reranking, query rewriting) based on features of the incoming query.

The system treats each RAG configuration as an **arm**, each query as a **context**, and the LLM-judged answer quality as the **reward signal**.

---

## Final Submission Contents

| File | Description |
|---|---|
| `final_project_jeetkava_devchira_vanshpra.pdf` (root) | Final NeurIPS-style technical report (also at `report/`) |
| `report/final_project_jeetkava_devchira_vanshpra.tex` | LaTeX source for the report |
| `report/diagrams/` | Mermaid sources and rendered PNGs for the 3 architecture diagrams |
| `notebooks/final_project_jeetkava_devchira_vanshpra.ipynb` | Final notebook with the head-to-head comparison results saved as cell outputs |
| `notebooks/final_project_checkpoint_jeetkava_devchira_vanshpra_FIXED.ipynb` | Earlier checkpoint notebook (baseline pipeline trace) |
| `bandits/`, `env/`, `rag/` | Source code (bandits, RL environment, RAG components) |
| `precompute_rewards.py`, `train_bandit.py`, `analyze_results.py`, `run_compare.py` | Top-level scripts |
| `slurm/` | UB CCR job scripts used for training |
| `results/task_0/bandit_results.npz` | Saved per-step bandit training data (5000 episodes, all 4 algorithms) |
| `results/compare_fair/` | Apples-to-apples comparison artifacts (10 queries, baseline vs RL) |

---

## Problem Formulation

| Component | Description |
|---|---|
| **State** | 11-dimensional vector: query length, keyword flags (list/steps/compare/why/math), BM25 top score, BM25 gap, dense top score, dense gap |
| **Actions** | 54 RAG configurations spanning: query rewrite (on/off), multi-query count (1/3, with `nq=1` forced when `rewrite=0`), reranking (on/off), final chunk count (5/10/20), BM25 weight (0.2/0.5/0.8) |
| **Reward** | Weighted sum of LLM-judged correctness, faithfulness, evidence recall, minus latency and token cost penalties |
| **Algorithms** | LinUCB Disjoint (contextual), Linear Thompson Sampling (contextual, used for final eval), UCB1 (baseline), ε-Greedy (baseline) |

---

## Project Structure

```
final-project-team_4/
├── bandits/                   # Bandit algorithm implementations
│   ├── base.py                # Abstract base class
│   ├── epsilon_greedy.py      # Epsilon-greedy (non-contextual)
│   ├── ucb.py                 # UCB1 (non-contextual)
│   ├── linucb.py              # LinUCB Disjoint (contextual)
│   └── thompson.py            # Linear Thompson Sampling (contextual, main)
├── env/                       # RL environment
│   ├── rag_env.py             # 1-step RAG environment with reward function
│   ├── action_space.py        # 54-arm action space
│   └── state_features.py      # 11-dim state feature extractor
├── rag/                       # RAG pipeline components
│   ├── pipeline.py            # End-to-end RAG pipeline
│   ├── config.py              # RAGConfig dataclass
│   ├── index_bm25.py          # BM25 retrieval
│   ├── index_dense.py         # Dense (Ollama embeddings) retrieval
│   ├── hybrid_rank.py         # Hybrid BM25 + dense fusion
│   ├── rewrite.py             # Query rewriting + multi-query expansion
│   ├── rerank.py              # Cross-encoder reranker
│   ├── llm.py                 # Ollama LLM wrapper
│   ├── judge.py               # LLM-as-judge scorer
│   └── metrics.py             # Correctness, faithfulness, recall metrics
├── data/
│   ├── prudentservices_dataset_rag.pdf    # Source document (security services)
│   ├── evalset_5_gold.jsonl               # 5-item gold eval set
│   ├── evalset_25_gold.jsonl              # 25-item gold eval set
│   └── evalset_100_gold.jsonl             # 100-item full eval set
├── slurm/                     # CCR HPC job scripts
│   ├── 0_precompute.sh        # Phase 1: compute reward table
│   ├── 1_merge.sh             # Phase 1b: merge partial reward files
│   └── 2_train_bandit.sh      # Phase 2: train all bandit algorithms
├── notebooks/
│   ├── final_project_jeetkava_devchira_vanshpra.ipynb            # Final notebook (with outputs)
│   └── final_project_checkpoint_jeetkava_devchira_vanshpra_FIXED.ipynb
├── report/                    # LaTeX report + diagrams
├── results/
│   ├── task_0/                # Saved bandit training (npz + summary)
│   ├── compare_fair/          # Final apples-to-apples comparison artifacts
│   └── baseline_5_*           # Initial 5-query baseline trace
├── precompute_rewards.py      # Phase 1 reward precomputation
├── train_bandit.py            # Phase 2 bandit training
├── analyze_results.py         # Results analysis
├── run_compare.py             # Final apples-to-apples baseline-vs-RL evaluation
└── requirements_ccr.txt       # Python dependencies
```

---

## Final Results (10-query apples-to-apples comparison)

Both the fixed baseline and the trained Thompson Sampling policy were run on the **same 10 queries** (6 in-domain, 4 out-of-domain) with the **same warm Ollama instance**. Per-query metrics are saved at `results/compare_fair/`.

| Metric | Baseline (mean) | RL+RAG Thompson (mean) | Per-query head-to-head (RL / Base / Ties) |
|---|---|---|---|
| Correctness (0–2) | 1.90 | 1.80 | 0 / 1 / 9  (tie) |
| Faithfulness (0–1) | 1.00 | 0.90 | 0 / 1 / 9  (tie) |
| Evidence recall@k (in-domain) | 0.167 | 0.167 | 1 / 1 / 4  (tie) |
| Total latency (s) | 9.36 | **4.51** | **9 / 1 / 0  (RL wins)** |
| Total tokens (mean / median) | 1160 / 1124 | 1230 / **944** | **6 / 4 / 0  (RL wins on per-query and median)** |
| Reward (scalar) | 0.860 | **1.024** | **9 / 1 / 0  (RL wins)** |

**Headline.** The trained policy wins clearly on the cost metrics (latency, reward, tokens by per-query and median count) while staying within noise on the quality metrics. The mean tokens go up because of one outlier query (`in_006`) where the policy gambled on a no-rerank, large-context configuration and lost on quality. On the other 9 queries RL is faster, cheaper or equal, and gets a higher scalar reward 9 out of 10 times.

Baseline config: `rewrite=True, nq=3, rerank=True, k_final=10, x_bm25=0.5`.

---

## Reproducibility

### Final apples-to-apples comparison
```bash
# Requires Ollama running with llama3.2:3b and nomic-embed-text pulled
python run_compare.py --bandit thompson \
    --evalset data/evalset_100_gold.jsonl \
    --ckpt_dir results/task_0 \
    --out_dir results/compare_fair
```
Outputs:
- `results/compare_fair/baseline_metrics.csv`
- `results/compare_fair/rl_thompson_metrics.csv`
- `results/compare_fair/details_thompson.jsonl`
- `results/compare_fair/summary_thompson.json`

### Bandit training (replay-style on saved reward table)
```bash
python train_bandit.py --algo all \
    --reward_table results/reward_table.npz \
    --out_dir results/task_0 \
    --episodes 5000 --seed 42
```

---

## Bandit Algorithms

| Algorithm | Type | Key Parameter |
|---|---|---|
| **LinUCB Disjoint** | Contextual | `--alpha` (exploration coefficient) |
| **Linear Thompson Sampling** | Contextual | `--v` (posterior variance scale) |
| **UCB1** | Non-contextual baseline | `--ucb_c` |
| **ε-Greedy** | Non-contextual baseline | `--epsilon`, `--epsilon_decay` |

LinUCB and Thompson Sampling use the 11-dimensional query state vector to learn per-arm linear reward models. UCB1 and ε-Greedy are context-free baselines.

---

## Running on UB CCR (training pipeline)

### Prerequisites
- UB CCR account with `general-compute` QOS access
- Ollama binary installed to `~/ollama_install/bin/ollama`
- Models pre-pulled: `llama3.2:3b` and `nomic-embed-text`

### Setup (one-time)

```bash
ssh UBIT@vortex.ccr.buffalo.edu
git clone https://github.com/ub-cse546-s26/final-project-team_4.git
cd final-project-team_4

module load gcc/11.2.0
module load python/3.9.6
module load cuda/11.8.0

python -m venv ~/envs/rl_rag
source ~/envs/rl_rag/bin/activate
pip install --upgrade pip
pip install -r requirements_ccr.txt

mkdir -p ~/ollama_install && cd ~/ollama_install
curl -L https://github.com/ollama/ollama/releases/download/v0.21.2/ollama-linux-amd64.tar.zst -o ollama-linux-amd64.tar.zst
tar --use-compress-program=unzstd -xf ollama-linux-amd64.tar.zst
echo 'export PATH="$HOME/ollama_install/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
cd ~/final-project-team_4

export OLLAMA_MODELS="$HOME/ollama_models" && mkdir -p "$OLLAMA_MODELS"
ollama serve &
sleep 8
ollama pull llama3.2:3b
ollama pull nomic-embed-text
kill %1

# If needed, edit --account/--qos directly inside slurm/*.sh for your CCR allocation.
```

### Submit Training Jobs

```bash
mkdir -p logs results/rewards checkpoints

# Phase 1: precompute reward table
sbatch slurm/0_precompute.sh
squeue -u $USER

# Phase 1b: merge partial reward files
sbatch slurm/1_merge.sh

# Phase 2: train all bandit algorithms
sbatch slurm/2_train_bandit.sh

# Analyze results
python analyze_results.py --results_dir results/
cat results/bandit_summary.json
```

---

## Dependencies

- Python 3.9.6
- `torch`, `transformers`, `sentence-transformers` — cross-encoder reranker
- `faiss-cpu` — dense vector index
- `rank-bm25` — BM25 retrieval
- `numpy`, `scipy`, `pandas`, `matplotlib` — bandit math + analysis
- `pypdf` — PDF ingestion
- `requests` — Ollama API calls
- Ollama (external) — LLM inference (`llama3.2:3b`, `nomic-embed-text`)

---

## References

The full bibliography is in the report. Tools used during development include ChatGPT (planning, draft text) and Cursor AI (in-editor code assistance). All RL design decisions, reward shaping, evalset construction, and result analysis were authored by the team.
