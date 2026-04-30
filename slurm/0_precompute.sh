#!/bin/bash
#SBATCH --job-name=rag_precompute
#SBATCH --output=logs/precompute_%A_%a.out
#SBATCH --error=logs/precompute_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=cse546s26
#SBATCH --gres=gpu:1
#SBATCH --array=0-9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jeetkava@buffalo.edu

EVALSET="data/evalset_100_gold.jsonl"
CHUNK_SIZE=10
OLLAMA_PORT=11434
OLLAMA_MODEL="llama3.2:3b"
EMBED_MODEL="nomic-embed-text"

export OLLAMA_MODELS="$HOME/ollama_models"
export OLLAMA_JUDGE_MODEL="llama3.2:3b"
export OLLAMA_ANSWER_MODEL="llama3.2:3b"
export OLLAMA_REWRITE_MODEL="llama3.2:3b"

set -euo pipefail
mkdir -p logs results/rewards

module purge
module load gcc/11.2.0
module load python/3.9.6
module load cuda/11.8.0

source $HOME/envs/rl_rag/bin/activate

if nvidia-smi > /dev/null 2>&1; then
    echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    export OLLAMA_GPU=1
else
    echo "No GPU - running on CPU"
    export CUDA_VISIBLE_DEVICES=""
fi

OLLAMA_BIN="$HOME/ollama_install/bin/ollama"

if [ ! -f "$OLLAMA_BIN" ]; then
    OLLAMA_BIN=$(which ollama 2>/dev/null || echo "")
fi

if [ -z "$OLLAMA_BIN" ]; then
    echo "ERROR: ollama binary not found." >&2
    exit 1
fi

export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"

echo "Starting Ollama server on port ${OLLAMA_PORT} ..."
"$OLLAMA_BIN" serve &
OLLAMA_PID=$!

echo "Ollama PID: $OLLAMA_PID"

for i in $(seq 1 60); do
    if curl -sf "http://${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        echo "Ollama ready after ${i}s"
        break
    fi
    sleep 1
done

START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))

echo "Task ${SLURM_ARRAY_TASK_ID}: items [${START}, ${END})"

cd $HOME/final-project-team_4

python precompute_rewards.py compute \
    --evalset     "$EVALSET" \
    --start       "$START" \
    --end         "$END" \
    --out_dir     "results/rewards" \
    --ollama_host "http://${OLLAMA_HOST}" \
    --embed_model "$EMBED_MODEL" \
    --llm_model   "$OLLAMA_MODEL"

kill "$OLLAMA_PID" 2>/dev/null || true

echo "Done."
ENDOFFILE

