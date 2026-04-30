# External Integrations
_Last updated: 2026-04-30_

## Summary

The project integrates exclusively with Ollama, a locally-hosted LLM inference server. All external "API" calls are made over HTTP to `localhost:11434`; there are no calls to cloud providers, remote databases, or third-party SaaS services. The one remote model dependency is a Hugging Face cross-encoder (`BAAI/bge-reranker-base`) downloaded via the `transformers` library at first use.

---

## APIs & External Services

### Ollama (Local HTTP Server)

The sole runtime service dependency. All communication uses plain HTTP POST to a locally running Ollama process.

**Text Generation (`rag/llm.py`):**
- Endpoint: `POST {OLLAMA_BASE_URL}/api/generate`
- Model: `llama3.2:3b` (configurable via `OLLAMA_ANSWER_MODEL` / `OLLAMA_CHAT_MODEL`)
- Used by: `OllamaLLM.generate()` — answer generation and LLM-as-judge scoring
- Auth: None (local loopback only)
- Config: `OLLAMA_BASE_URL` env var (default `http://localhost:11434`)

**Embeddings (`rag/index_dense.py`):**
- Endpoint: `POST {OLLAMA_BASE_URL}/api/embed`
- Model: `nomic-embed-text` (configurable via `OLLAMA_EMBED_MODEL`)
- Used by: `DenseIndex._embed_batch()` — document chunk and query embeddings
- Auth: None (local loopback only)
- Config: `OLLAMA_BASE_URL`, `OLLAMA_EMBED_MODEL` env vars

**Readiness Check (`slurm/0_precompute.sh`):**
- Endpoint: `GET {OLLAMA_HOST}/api/tags`
- Used by: SLURM precompute script to poll until Ollama server is ready (up to 60s)

**Ollama Binary Installation:**
- Downloaded from: `https://github.com/ollama/ollama/releases/download/v0.21.2/ollama-linux-amd64.tar.zst`
- Installed to: `~/ollama_install/bin/ollama`
- Version pinned: `v0.21.2`

---

## Model Downloads (Hugging Face Hub)

### Cross-Encoder Reranker (`rag/rerank.py`)

- Model: `BAAI/bge-reranker-base`
- Downloaded via: `transformers.AutoModelForSequenceClassification.from_pretrained()`
- Download location: default Hugging Face cache (`~/.cache/huggingface/`)
- Trigger: lazy-loaded on first `CrossEncoderReranker.rerank()` call
- Internet required: Yes, on first use (cached thereafter)
- Auth: None (public model)

---

## Data Storage

**Local Filesystem Only — no external database.**

| Path | Format | Purpose |
|------|--------|---------|
| `data/prudentservices_dataset_rag.pdf` | PDF | Source document ingested by `rag/pdf_ingest.py` |
| `data/evalset_5_gold.jsonl` | JSONL | 5-item gold evaluation set |
| `data/evalset_25_gold.jsonl` | JSONL | 25-item gold evaluation set |
| `data/evalset_100_gold.jsonl` | JSONL | 100-item full evaluation set |
| `data/cache/emb_<model>_<hash>.npy` | NumPy `.npy` | Cached document embeddings (keyed by model + chunk hash) |
| `results/rewards/partial_<N>.npz` | NumPy `.npz` | Per-SLURM-task partial reward arrays |
| `results/reward_table.npz` | NumPy `.npz` | Merged reward table: shape `[n_queries, n_actions]` |
| `results/bandit_results.npz` | NumPy `.npz` | Per-step reward histories and regret arrays |
| `results/bandit_summary.json` | JSON | Final metrics table across algorithms |
| `checkpoints/<algo>.pkl` | Python pickle | Saved bandit agent state (LinUCB `A`/`b` matrices, etc.) |

**File Storage:** Local filesystem only.
**Caching:** Embedding cache in `data/cache/` (SHA-256 keyed `.npy` files, managed by `DenseIndex.build()`).
**Databases:** None.
**Message Queues:** None.

---

## Authentication & Identity

**None.** The project has no user authentication, API keys, OAuth flows, or secrets management. All service calls are to `localhost`.

---

## Monitoring & Observability

**Error Tracking:** None (no Sentry, Datadog, or equivalent).
**Logs:** SLURM stdout/stderr redirected to `logs/precompute_<JOBID>_<TASKID>.out/.err` and `logs/train_<JOBID>.out/.err`. Application-level logging uses `print()` statements throughout source files.
**Metrics:** `results/bandit_summary.json` and `results/baseline_5_metrics.json` are written at run completion.

---

## CI/CD & Deployment

**Hosting:** UB CCR HPC (`vortex.ccr.buffalo.edu`), SLURM `general-compute` partition.
**CI Pipeline:** None detected (no `.github/workflows/`, no CircleCI, no Jenkins config).
**Deployment model:** Manual SLURM job submission (`sbatch slurm/*.sh`).

---

## Environment Configuration

**Required environment variables at runtime:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_ANSWER_MODEL` | `hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF` | LLM for answer generation |
| `OLLAMA_CHAT_MODEL` | (fallback for answer model) | Alias for answer model |
| `OLLAMA_JUDGE_MODEL` | (falls back to OLLAMA_CHAT_MODEL) | LLM for judge scoring |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_MODELS` | `$HOME/ollama_models` | Ollama model weights directory |
| `OLLAMA_HOST` | Set by SLURM scripts to `127.0.0.1:<PORT>` | Ollama bind address |

**No `.env` file detected.** All variables are injected via SLURM job scripts or shell environment.

---

## Webhooks & Callbacks

**Incoming:** None.
**Outgoing:** None.

---

## Gaps & Unknowns

- The `sentence-transformers>=2.3` package is declared in `requirements_ccr.txt` but no direct usage of `SentenceTransformer` classes was found in source files — it may be an indirect dependency of the cross-encoder pipeline or reserved for future use.
- The default fallback LLM model name in `rag/llm.py` (`hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF`) differs from the SLURM-configured model (`llama3.2:3b`); the actual model used depends on which environment variable is set at runtime.
- Hugging Face Hub access requires outbound internet on first run to download `BAAI/bge-reranker-base`; CCR compute nodes may require proxy configuration for this.
- No requirements file exists for local (non-CCR) development; Python version and exact package versions for macOS/Linux dev environments are undocumented.

---

*Integration audit: 2026-04-30*
