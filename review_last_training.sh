#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/paper_reproductions/posture_control
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate landing-drqv2
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
run="$(cat runs/current_live_training.txt)"
exec python -m landing_repro.review --run "$run" --episodes 3
