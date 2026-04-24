#!/bin/bash
# =============================================================================
# SLURM: Merge partial reward .npz files after precompute array job finishes.
# Run this after 0_precompute.sh completes all array tasks.
#
# Submit with:
#   sbatch slurm/1_merge.sh
# =============================================================================

#SBATCH --job-name=rag_merge
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=introccr

set -euo pipefail
mkdir -p logs

module purge
module load gcc/11.2.0
module load python/3.9.6

source $HOME/envs/rl_rag/bin/activate

cd $HOME/final-project-team_4

python precompute_rewards.py merge \
    --out_dir   results/rewards \
    --merge_out results/reward_table.npz

echo "Merge done. Reward table at results/reward_table.npz"
