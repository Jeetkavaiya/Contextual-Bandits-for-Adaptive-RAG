# RL-RAG: Contextual Bandit Optimization for RAG Pipelines

**CSE 546 — Reinforcement Learning | University at Buffalo | Spring 2026**

**Team 4:** Jeet Kava · Dev Desai (devchira) · Vansh Pra

---

## Overview

This project applies **contextual multi-armed bandit algorithms** to optimize Retrieval-Augmented Generation (RAG) pipelines. Instead of using a fixed RAG configuration for every query, a bandit agent learns to select the best pipeline configuration (chunk count, hybrid retrieval weights, reranking, query rewriting) based on features of the incoming query.

The system treats each RAG configuration as an **arm**, each query as a **context**, and the LLM-judged answer quality as the **reward signal**.

---

## Problem Formulation

| Component | Description |
|---|---|
| **State** | 11-dimensional vector: query length, keyword flags (list/steps/compare/why/math), BM25 top score, BM25 gap, dense top score, dense gap |
| **Actions** | 60 RAG configurations spanning: query rewrite (on/off), multi-query count (1/3), reranking (on/off), final chunk count (5/10/20), BM25 weight (0.2/0.5/0.8) |
| **Reward** | Weighted sum of LLM-judged correctness, faithfulness, evidence recall, minus latency and token cost penalties |
| **Algorithms** | LinUCB (contextual), Thompson Sampling (contextual), UCB1 (baseline), ε-Greedy (baseline) |

---

## Project Structure

```
final-project-team_4/
├── bandits/                  # Bandit algorithm implementations
│   ├── base.py               # Abstract base class
│   ├── epsilon_greedy.py     # ε-Greedy (non-contextual baseline)
│   ├── ucb.py                # UCB1 (non-contextual baseline)
│   ├── linucb.py             # LinUCB Disjoint (contextual, main)
│   └── thompson.py           # Linear Thompson Sampling (contextual)
├── env/                      # RL environment
│   ├── rag_env.py            # OpenAI-gym-style RAG environment
│   ├── action_space.py       # 60-arm action space definition
│   └── state_features.py     # 11-dim state feature extractor
├── rag/                      # RAG pipeline components
│   ├── pipeline.py           # Main RAG pipeline
│   ├── config.py             # RAGConfig dataclass
│   ├── index_bm25.py         # BM25 retrieval
│   ├── index_dense.py        # Dense (Ollama embeddings) retrieval
│   ├── hybrid_rank.py        # Hybrid BM25 + dense fusion
│   ├── rewrite.py            # Query rewriting + multi-query expansion
│   ├── rerank.py             # Cross-encoder reranker
│   ├── llm.py                # Ollama LLM wrapper
│   ├── judge.py              # LLM-as-judge scorer
│   └── metrics.py            # Correctness, faithfulness, recall metrics
├── data/
│   ├── prudentservices_dataset_rag.pdf   # Source document (security services)
│   ├── evalset_5_gold.jsonl              # 5-item gold eval set
│   ├── evalset_25_gold.jsonl             # 25-item gold eval set
│   └── evalset_100_gold.jsonl            # 100-item full eval set
├── slurm/                    # CCR HPC job scripts
│   ├── 0_precompute.sh       # Phase 1: compute reward table (GPU/CPU)
│   ├── 1_merge.sh            # Phase 1b: merge partial reward files
│   └── 2_train_bandit.sh     # Phase 2: train all bandit algorithms
├── notebooks/
│   └── final_project_checkpoint_jeetkava_devchira_vanshpra_FIXED.ipynb
├── precompute_rewards.py     # Phase 1 reward precomputation script
├── train_bandit.py           # Phase 2 bandit training script
├── analyze_results.py        # Results analysis and plotting
└── requirements_ccr.txt      # Python dependencies for CCR
```

---

## Training Pipeline

Training runs in **3 phases** on UB CCR HPC:

```
Phase 1 → Phase 1b → Phase 2
```

### Phase 1 — Precompute Reward Table
Runs every eval item through all 60 RAG configurations using Ollama. Scores each (query, action) pair with the LLM judge. Saves partial results as `.npz` files. This is a SLURM array job (10 tasks × 10 items each).

### Phase 1b — Merge
Merges the 10 partial reward files into a single `results/reward_table.npz`.

### Phase 2 — Bandit Training
Loads the precomputed reward table and trains all 4 bandit algorithms. Outputs per-step reward histories, cumulative regret curves, and saved checkpoints.

---

## Baseline Results (5-item eval set)

| Metric | All | In-Domain | Out-of-Domain |
|---|---|---|---|
| Avg Correctness (0–2) | 1.80 | 1.67 | 2.00 |
| Avg Faithfulness (0–1) | 1.00 | 1.00 | 1.00 |
| Avg Evidence Recall | 0.33 | 0.33 | — |
| Avg Latency (s) | 214.6 | 204.0 | 230.4 |
| Avg Total Tokens | 1175.6 | 1099.3 | 1290.0 |

Baseline config: `rewrite=True, nq=3, rerank=True, k_final=10, x_bm25=0.5`

---

## Running on UB CCR

### Prerequisites
- UB CCR account with `general-compute` QOS access
- Ollama binary installed to `~/ollama_install/bin/ollama`
- Models pre-pulled: `llama3.2:3b` and `nomic-embed-text`

### Setup (one-time)

```bash
# SSH in
ssh UBIT@vortex.ccr.buffalo.edu

# Clone repo
git clone https://github.com/ub-cse546-s26/final-project-team_4.git
cd final-project-team_4

# Load modules
module load gcc/11.2.0
module load python/3.9.6
module load cuda/11.8.0

# Create venv
python -m venv ~/envs/rl_rag
source ~/envs/rl_rag/bin/activate
pip install --upgrade pip
pip install -r requirements_ccr.txt

# Install Ollama
mkdir -p ~/ollama_install && cd ~/ollama_install
curl -L https://github.com/ollama/ollama/releases/download/v0.21.2/ollama-linux-amd64.tar.zst -o ollama-linux-amd64.tar.zst
tar --use-compress-program=unzstd -xf ollama-linux-amd64.tar.zst
echo 'export PATH="$HOME/ollama_install/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
cd ~/final-project-team_4

# Pull models
export OLLAMA_MODELS="$HOME/ollama_models" && mkdir -p "$OLLAMA_MODELS"
ollama serve &
sleep 8
ollama pull llama3.2:3b
ollama pull nomic-embed-text
kill %1

# Update SLURM scripts with your account (check with: sacctmgr show user UBIT withassoc format=account,partition,qos -p)
sed -i 's|--account=introccr|--account=YOUR_ACCOUNT|g' slurm/*.sh
sed -i 's|--qos=general-compute|--qos=YOUR_QOS|g' slurm/*.sh
```

### Submit Training Jobs

```bash
# Phase 1 — precompute rewards (needs GPU for speed)
mkdir -p logs results/rewards checkpoints
sbatch slurm/0_precompute.sh
squeue -u $USER          # monitor — wait until all 10 tasks finish

# Phase 1b — merge reward files
sbatch slurm/1_merge.sh
squeue -u $USER          # wait until done
ls -lh results/reward_table.npz   # confirm

# Phase 2 — train bandits
sbatch slurm/2_train_bandit.sh
squeue -u $USER          # wait until done

# Analyze results
source ~/envs/rl_rag/bin/activate
python analyze_results.py --results_dir results/
cat results/bandit_summary.json
```

### Monitor Live Output

```bash
tail -f logs/precompute_JOBID_0.out   # Phase 1 live log
tail -f logs/train_JOBID.out          # Phase 2 live log
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

## Dependencies

- Python 3.9.6
- `torch`, `transformers`, `sentence-transformers` — cross-encoder reranker
- `faiss-cpu` — dense vector index
- `rank-bm25` — BM25 retrieval
- `numpy`, `scipy` — bandit math
- `pypdf` — PDF ingestion
- `requests` — Ollama API calls
- Ollama (external) — LLM inference (`llama3.2:3b`, `nomic-embed-text`)
