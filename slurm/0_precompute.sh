#!/bin/bash
# =============================================================================
# SLURM: Phase 1 — Pre-compute reward table (needs Ollama + GPU for speed)
#
# This is a SLURM array job.  Each task handles a slice of eval items.
# Adjust CHUNK_SIZE and the evalset path to your needs.
#
# Submit with:
#   sbatch slurm/0_precompute.sh
#
# After all tasks finish, run merge:
#   python precompute_rewards.py merge --out_dir results/rewards \
#       --merge_out results/reward_table.npz
# =============================================================================

#SBATCH --job-name=rag_precompute
#SBATCH --output=logs/precompute_%A_%a.out
#SBATCH --error=logs/precompute_%A_%a.err
#SBATCH --time=04:00:00          # 4 hours per item-slice (60 actions × ~30s each)
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1             # GPU for faster cross-encoder reranking + Ollama
#SBATCH --partition=general-compute
#SBATCH --array=0-9              # 10 tasks × 10 items each = 100 items total
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=devchira@buffalo.edu

# ──────────────────────────────────────────────────────────────────────────────
# Editable config
# ──────────────────────────────────────────────────────────────────────────────
EVALSET="data/evalset_100_gold.jsonl"
CHUNK_SIZE=10           # items per array task  (array 0-9 → items 0-99)
OLLAMA_PORT=11434
OLLAMA_MODEL="llama3.2:3b"
EMBED_MODEL="nomic-embed-text"

# Where Ollama stores downloaded models (must be pre-pulled on CCR shared FS)
export OLLAMA_MODELS="/projects/academic/YOUR_GROUP/ollama_models"  # <-- change
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs results/rewards

# ── 1. Load modules ───────────────────────────────────────────────────────────
module purge
module load gcc/11.2.0
module load python/3.11.3           # or: module load anaconda3/2023.09-0
module load cuda/12.1               # needed for torch / FAISS

# ── 2. Activate conda env ─────────────────────────────────────────────────────
# If using conda:
# source activate rl_rag
# If using venv:
source /projects/academic/YOUR_GROUP/envs/rl_rag/bin/activate  # <-- change

# ── 3. Start Ollama server in background ──────────────────────────────────────
# Ollama binary should already be in PATH or set OLLAMA_BIN
OLLAMA_BIN="${OLLAMA_MODELS}/../ollama"
if [ ! -f "$OLLAMA_BIN" ]; then
    OLLAMA_BIN=$(which ollama 2>/dev/null || echo "")
fi

if [ -z "$OLLAMA_BIN" ]; then
    echo "ERROR: ollama binary not found. Set OLLAMA_BIN or add to PATH." >&2
    exit 1
fi

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
echo "Starting Ollama server on port ${OLLAMA_PORT} …"
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait for Ollama to be ready (retry up to 60s)
for i in $(seq 1 60); do
    if curl -sf "http://${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        echo "Ollama ready after ${i}s"
        break
    fi
    sleep 1
done

# ── 4. Compute item slice ─────────────────────────────────────────────────────
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))

echo "Task ${SLURM_ARRAY_TASK_ID}: items [${START}, ${END})"

cd /projects/academic/YOUR_GROUP/RL/final-project-team_4  # <-- change REPO ROOT

python precompute_rewards.py compute \
    --evalset     "$EVALSET" \
    --start       "$START" \
    --end         "$END" \
    --out_dir     "results/rewards" \
    --ollama_host "http://${OLLAMA_HOST}" \
    --embed_model "$EMBED_MODEL" \
    --llm_model   "$OLLAMA_MODEL"

# ── 5. Cleanup ────────────────────────────────────────────────────────────────
kill "$OLLAMA_PID" 2>/dev/null || true
echo "Done."
