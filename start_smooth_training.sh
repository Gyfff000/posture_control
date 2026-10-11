#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f runs/current_smooth_training.txt ]]; then
    previous="$(cat runs/current_smooth_training.txt)"
    if [[ -f "$previous/training.pid" ]]; then
        previous_pid="$(cat "$previous/training.pid")"
        if [[ -r "/proc/$previous_pid/cmdline" ]] && grep -aq 'landing_repro.smooth' "/proc/$previous_pid/cmdline"; then
            printf 'Smooth visual training is already running (PID %s).\n' "$previous_pid"
            exit 0
        fi
    fi
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate landing-drqv2
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
checkpoint="${1:-runs/smooth_prepare_seed31/initial.pt}"
out="runs/smooth_long_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
printf '%s\n' "$out" > runs/current_smooth_training.txt
printf '%s\n' "$$" > "$out/training.pid"
exec python -m landing_repro.smooth train --checkpoint "$checkpoint" --data runs/smooth_prepare_seed31/demonstrations.npz --output "$out" --stage 1 --steps 60000 --eval-every 5000 --eval-episodes 20 --seed 41 --prediction-guidance --gui
