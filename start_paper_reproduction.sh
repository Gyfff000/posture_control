#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate landing-drqv2
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
pointer=runs/current_paper_training.txt
if [[ -f "$pointer" ]]; then
    out="$(cat "$pointer")"
    if [[ -f "$out/training.pid" ]]; then
        pid="$(cat "$out/training.pid")"
        if [[ -r "/proc/$pid/cmdline" ]] && grep -aq 'paper_repro.train' "/proc/$pid/cmdline"; then
            printf '%s\n' "$(date +%s)" > "$out/show_window.request"
            printf 'Paper-derived training already running (PID %s). Display requested.\n' "$pid"
            exit 0
        fi
    fi
    if [[ -f "$out/latest.pt" ]]; then
        exec python -m paper_repro.train --output "$out" --seed 1 --variant baseline --reward-interpretation normalized --vertical-reward actual_guard --gui --resume
    fi
fi
out="runs/paper_ladosz2024_guarded_seed1_$(date +%Y%m%d_%H%M%S)"
printf '%s\n' "$out" > "$pointer"
exec python -m paper_repro.train --output "$out" --seed 1 --variant baseline --reward-interpretation normalized --vertical-reward actual_guard --gui
