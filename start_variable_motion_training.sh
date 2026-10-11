#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
previous="$(cat runs/current_smooth_training.txt)"
if [[ -f "$previous/training.pid" ]]; then
    previous_pid="$(cat "$previous/training.pid")"
    if [[ -r "/proc/$previous_pid/cmdline" ]] && grep -aq 'landing_repro.smooth' "/proc/$previous_pid/cmdline"; then
        printf 'Existing training or completed desktop window is still open (PID %s). Close it first.\n' "$previous_pid"
        exit 0
    fi
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate landing-drqv2
export LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
checkpoint="$previous/latest.pt"
if [[ -f "$previous/interrupted.pt" ]]; then checkpoint="$previous/interrupted.pt"; fi
checkpoint="${1:-$checkpoint}"
if [[ ! -f "$checkpoint" ]]; then printf 'Checkpoint not found: %s\n' "$checkpoint"; exit 1; fi
saved_steps="$(python -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["steps"])' "$checkpoint")"
total_steps=$((saved_steps+80000))
out="runs/motion_curriculum_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
printf '%s\n' "$out" > runs/current_smooth_training.txt
printf '%s\n' "$$" > "$out/training.pid"
exec python -m landing_repro.smooth train --checkpoint "$checkpoint" \
    --data runs/smooth_prepare_seed31/demonstrations.npz --output "$out" \
    --motion-curriculum --steps "$total_steps" --step-offset "$saved_steps" \
    --resume-run "$previous" --eval-every 5000 --eval-episodes 20 --seed 41 \
    --prediction-guidance --near-landing-assist --terminal-finish-assist --gui
