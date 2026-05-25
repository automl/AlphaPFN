#!/bin/bash
# slurm wrapper for the PPD base-model training recipe.
#
# Usage:   sbatch model_training/slurm/slurm_train_ppd.sh [SEED]
# Default: SEED=0
#
# The partition and time limit mirror the kislurm shell script that
# produced the v1 PPD checkpoint. Reviewers without slurm can run
# model_training/train.py directly (torchrun for multi-GPU, plain python for 1 GPU).
#
#SBATCH -p mldlc2_gpu-l40s
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --job-name=alphapfn_train_ppd
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

SEED=${1:-0}
ALPHAPFN=${ALPHAPFN:-$(cd "$(dirname "$0")/../.." && pwd)}
OUTPUT_DIR=${OUTPUT_DIR:-"$ALPHAPFN/final_models"}

cd "$ALPHAPFN"

echo "host: $(hostname)"
echo "slurm job: $SLURM_JOB_ID  gpus: $SLURM_GPUS_ON_NODE"
echo "alphapfn:  $ALPHAPFN"
echo "output:    $OUTPUT_DIR"
echo "seed:      $SEED"

# Use torchrun for multi-GPU. init_dist() reads $LOCAL_RANK, set by torchrun.
torchrun --nproc-per-node=4 model_training/train.py \
    --acquisition ppd \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"
