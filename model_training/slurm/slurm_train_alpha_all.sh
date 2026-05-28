#!/bin/bash
# Submit all 3 direct-head training jobs (PES, MES, JES) sequentially.
# Each submission is independent and will queue on alldlc2_gpu-l40s.
#
# Usage:   bash model_training/slurm/slurm_train_alpha_all.sh [SEED]
# Default SEED=0.

set -euo pipefail

SEED=${1:-0}
HERE=$(cd "$(dirname "$0")" && pwd)

for af in pes mes jes; do
    sbatch "$HERE/slurm_train_alpha.sh" "$af" "$SEED"
done
