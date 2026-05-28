#!/bin/bash
# slurm wrapper for direct-head (PES/MES/JES) training recipes.
#
# Usage:   sbatch model_training/slurm/slurm_train_alpha.sh ACQUISITION [SEED]
#   ACQUISITION ∈ {pes, mes, jes}
# Default SEED=0.
#
# Partition is alldlc2_gpu-l40s to match the kislurm direct-head recipe
# (different from the PPD recipe which used mldlc2_gpu-h200).
#
#SBATCH -p alldlc2_gpu-l40s
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --job-name=alphapfn_train_alpha
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ACQUISITION=${1:?usage: slurm_train_alpha.sh ACQUISITION [SEED] — ACQUISITION ∈ {pes, mes, jes}}
SEED=${2:-0}

case "$ACQUISITION" in
    pes|mes|jes) ;;
    *) echo "error: ACQUISITION must be one of pes/mes/jes, got '$ACQUISITION'" >&2; exit 2 ;;
esac

ALPHAPFN=${ALPHAPFN:-$(cd "$(dirname "$0")/../.." && pwd)}
OUTPUT_DIR=${OUTPUT_DIR:-"$ALPHAPFN/final_models"}

cd "$ALPHAPFN"

echo "host: $(hostname)"
echo "slurm job: $SLURM_JOB_ID  gpus: $SLURM_GPUS_ON_NODE"
echo "alphapfn:  $ALPHAPFN"
echo "output:    $OUTPUT_DIR"
echo "acquisition: $ACQUISITION  seed: $SEED"

torchrun --nproc-per-node=4 model_training/train.py \
    --acquisition "$ACQUISITION" \
    --seed "$SEED" \
    --output-dir "$OUTPUT_DIR"
