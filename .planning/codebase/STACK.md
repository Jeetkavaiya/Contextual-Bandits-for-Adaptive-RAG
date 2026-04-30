# Technology Stack
_Last updated: 2026-04-30_

## Summary

This project is a Python 3.9 research codebase that implements contextual multi-armed bandit algorithms to optimize RAG (Retrieval-Augmented Generation) pipelines. It combines classical information retrieval (BM25, dense embeddings), transformer-based reranking, and local LLM inference via Ollama, trained on UB CCR HPC using SLURM. The bandit training phase is pure NumPy and runs on CPU; the reward-precomputation phase requires GPU-accelerated Ollama inference.

---

## Languages

**Primary:**
- Python 3.9.6 — all source code (`bandits/`, `rag/`, `env/`, entry-point scripts)

**Secondary:**
- Bash — SLURM job scripts (`slurm/0_precompute.sh`, `slurm/1_merge.sh`, `slurm/2_train_bandit.sh`)
- Jupyter Notebook — exploratory work (`notebooks/final_project_checkpoint_jeetkava_devchira_vanshpra_FIXED.ipynb`, `notebooks/test.ipynb`)

---

## Runtime

**Environment:**
- CPython 3.9.6 (pinned for UB CCR compatibility)
- Virtual environment managed via `python -m venv ~/envs/rl_rag`

**Package Manager:**
- pip (no lockfile; only `requirements_ccr.txt` with `>=` version bounds)
- Lockfile: absent

---

## Frameworks

**Core Scientific Stack:**
- `numpy>=1.24` — bandit math (LinUCB ridge regression, reward table storage as `.npz`), embedding arrays
- `scipy>=1.10` — supporting scientific utilities

**Machine Learning / NLP:**
- `torch>=2.0` — PyTorch, used exclusively by the cross-encoder reranker (`rag/rerank.py`)
- `transformers>=4.35` — Hugging Face Transformers; loads `BAAI/bge-reranker-base` cross-encoder via `AutoModelForSequenceClassification` / `AutoTokenizer` (`rag/rerank.py`)
- `sentence-transformers>=2.3` — declared dependency, available for embedding utilities (primary embeddings use Ollama API directly)

**Information Retrieval:**
- `rank-bm25>=0.2.2` — `BM25Okapi` sparse retrieval index (`rag/index_bm25.py`)
- `faiss-cpu>=1.7.4` — dense vector search with `IndexFlatIP` (cosine similarity via L2-normalized inner product) (`rag/index_dense.py`)

**Document Ingestion:**
- `pypdf>=3.0` — PDF text extraction from `data/prudentservices_dataset_rag.pdf` (`rag/pdf_ingest.py`)

**HTTP Client:**
- `requests>=2.31` — all Ollama REST API calls (embeddings at `/api/embed`, generation at `/api/generate`) (`rag/llm.py`, `rag/index_dense.py`)

**Analysis / Visualization:**
- `matplotlib>=3.7` — plotting regret curves and reward distributions (`analyze_results.py`)
- `pandas>=2.0` — tabular results processing (`analyze_results.py`)

**Build/Dev:**
- None detected (no pytest, no linter config, no formatter config)

---

## Key Dependencies

**Critical:**
- `numpy` — reward table (`.npz` format), all bandit algorithm math (LinUCB, Thompson Sampling, UCB1, ε-Greedy)
- `faiss-cpu` — dense vector index; without it, dense retrieval and embedding caching are unavailable
- `rank-bm25` — BM25 sparse retrieval; without it, the BM25 index cannot be built
- `torch` + `transformers` — cross-encoder reranker `BAAI/bge-reranker-base`; lazy-loaded on first reranker call
- `requests` — Ollama API bridge; required for both LLM generation and embedding

**Infrastructure:**
- `pypdf` — one-time PDF ingestion to produce `Chunk` objects
- `scipy` — auxiliary math (not central but declared)

---

## Configuration

**Environment Variables (runtime, no `.env` file detected):**
- `OLLAMA_BASE_URL` — Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_ANSWER_MODEL` / `OLLAMA_CHAT_MODEL` — LLM model name (default: `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF`)
- `OLLAMA_EMBED_MODEL` — embedding model name (default: `nomic-embed-text`)
- `OLLAMA_JUDGE_MODEL` — judge LLM (falls back to `OLLAMA_CHAT_MODEL`)
- `OLLAMA_MODELS` — filesystem path where Ollama stores downloaded model weights (set to `$HOME/ollama_models` in SLURM scripts)
- `OLLAMA_GPU` — set to `1` when GPU is detected in SLURM precompute job

**Build Config:**
- `requirements_ccr.txt` — sole dependency manifest; no `setup.py`, `pyproject.toml`, or `Pipfile`

**Data Artifacts:**
- `data/cache/` — NumPy `.npy` embedding cache files (keyed by model name + chunk hash)
- `results/reward_table.npz` — merged precomputed reward table (NumPy compressed archive)
- `checkpoints/*.pkl` — serialized bandit agent state (pickle)

---

## Platform Requirements

**Development / Local:**
- Python 3.9+, pip
- Ollama binary running locally (serves LLM inference and embeddings)
- Models pre-pulled: `llama3.2:3b`, `nomic-embed-text`
- GPU optional (speeds up reranker and Ollama inference)

**Production / HPC (UB CCR):**
- SLURM scheduler on `general-compute` partition / QOS
- Modules: `gcc/11.2.0`, `python/3.9.6`, `cuda/11.8.0`
- GPU node (`--gres=gpu:1`) for Phase 1 (reward precomputation)
- CPU-only node sufficient for Phase 2 (bandit training)
- 32 GB RAM for Phase 1; 16 GB RAM for Phase 2
- Ollama v0.21.2 binary installed at `~/ollama_install/bin/ollama`

---

*Stack analysis: 2026-04-30*
