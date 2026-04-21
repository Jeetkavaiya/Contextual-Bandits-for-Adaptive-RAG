#!/bin/bash
# =============================================================================
# SLURM: Phase 2 — Train bandit algorithms on pre-computed reward table.
#
# Pure NumPy — no Ollama, no GPU required.  Runs fast on any CPU node.
# Submit AFTER reward_table.npz exists (after jobs 0 + 1).
#
# Submit with:
#   sbatch slurm/2_train_bandit.sh
#
# Optional: run a hyperparameter sweep by submitting an array job:
#   sbatch --array=0-7 slurm/2_train_bandit.sh
#   (uses SLURM_ARRAY_TASK_ID to select alpha / v from ALPHA_LIST / V_LIST)
# =============================================================================

#SBATCH --job-name=rag_bandit_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general-compute
# Remove the GPU line — not needed for bandit training:
# #SBATCH --gres=gpu:1

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter sweep (array job mode)
# Each array task picks a different alpha/v combination.
# ──────────────────────────────────────────────────────────────────────────────
ALPHA_LIST=(0.1 0.3 0.5 1.0 1.5 2.0 3.0 5.0)
V_LIST=(    0.1 0.3 0.5 1.0 1.5 2.0 3.0 5.0)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ALPHA=${ALPHA_LIST[$TASK_ID]}
V_VAL=${V_LIST[$TASK_ID]}

# If not an array job, use defaults
ALPHA=${ALPHA:-1.0}
V_VAL=${V_VAL:-1.0}

# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
mkdir -p logs checkpoints results

module purge
module load gcc/11.2.0
module load python/3.11.3

source /projects/academic/YOUR_GROUP/envs/rl_rag/bin/activate  # <-- change

cd /projects/academic/YOUR_GROUP/RL/final-project-team_4  # <-- change

echo "Task ${TASK_ID}: alpha=${ALPHA}  v=${V_VAL}"

python train_bandit.py \
    --reward_table  results/reward_table.npz \
    --n_episodes    5000 \
    --seed          42 \
    --alpha         "$ALPHA" \
    --v             "$V_VAL" \
    --reg_lambda    1.0 \
    --epsilon       0.15 \
    --epsilon_decay 0.001 \
    --ucb_c         1.0 \
    --out_dir       "results/task_${TASK_ID}" \
    --checkpoint_dir "checkpoints/task_${TASK_ID}"

echo "Training complete for task ${TASK_ID}."
