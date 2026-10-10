#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f runs/current_guided_suite.txt ]]; then
    previous="$(cat runs/current_guided_suite.txt)"
    if [[ -f "$previous/suite.pid" ]]; then
        previous_pid="$(cat "$previous/suite.pid")"
        if [[ -r "/proc/$previous_pid/cmdline" ]] && grep -aq 'landing_repro.guided_suite' "/proc/$previous_pid/cmdline"; then
            printf 'Improved training suite is already running (PID %s).\n' "$previous_pid"
            exit 0
        fi
    fi
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate landing-drqv2
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
out="runs/guided_long_$(date +%Y%m%d_%H%M%S)"
initial="${1:-runs/guided_pretrain_seed1/latest.pt}"
mkdir -p "$out"
printf '%s\n' "$out" > runs/current_guided_suite.txt
printf '%s\n' "$$" > "$out/suite.pid"
exec python -m landing_repro.guided_suite --checkpoint "$initial" --data runs/guided_pretrain_seed1/demonstrations.npz --output "$out" --steps 60000 --seeds 11 22 --gui
