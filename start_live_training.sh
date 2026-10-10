#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/paper_reproductions/posture_control
if [[ -f runs/current_live_training.txt ]]; then
    previous="$(cat runs/current_live_training.txt)"
    if [[ -f "$previous/training.pid" ]]; then
        previous_pid="$(cat "$previous/training.pid")"
        if [[ -r "/proc/$previous_pid/cmdline" ]] && grep -aq 'landing_repro.train' "/proc/$previous_pid/cmdline"; then
            printf 'Training is already running (PID %s). Use its Pause / Stop buttons.\n' "$previous_pid"
            exit 0
        fi
    fi
fi
python3 - <<'PY'
from pathlib import Path
import os, signal
for f in [Path('runs/desktop_oracle/demo.pid'), Path('runs/desktop_drq/demo.pid')]:
    if f.exists():
        try:
            pid = int(f.read_text().strip())
            cmd = Path('/proc', str(pid), 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if 'landing_repro.demo' in cmd:
                os.kill(pid, signal.SIGTERM)
                print('Stopped old evaluation demo', pid)
        except (ValueError, FileNotFoundError, ProcessLookupError):
            pass
PY
out="runs/live_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
printf '%s\n' "$out" > runs/current_live_training.txt
printf '%s\n' "$$" > "$out/training.pid"
printf 'Live training folder: %s\n' "$out"
exec bash ./run_train_cpu.sh --gui --realtime --steps 12000 --output "$out" > >(tee "$out/console.log") 2>&1
