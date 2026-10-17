#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate landing-drqv2
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
run="$(cat runs/current_paper_training.txt)"
test -f "$run/latest.pt"
out="results/evaluations/frozen_$(date +%Y%m%d_%H%M%S)_$$"
exec python -m paper_repro.evaluate --device cpu --checkpoint "$run/latest.pt" --output "$out" --episodes 3 --seed-start 901001 --gui
