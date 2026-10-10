#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate landing-drqv2
export LIBGL_ALWAYS_SOFTWARE=1
export LP_NUM_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
exec python -m landing_repro.demo "$@"
