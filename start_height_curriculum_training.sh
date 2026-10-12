#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export LANDING_HEIGHT_CURRICULUM=1
python_bin="$HOME/miniforge3/envs/landing-drqv2/bin/python"
previous="$(cat runs/current_smooth_training.txt)"
checkpoint="$previous/latest.pt"
if [[ -f "$previous/interrupted.pt" ]]; then checkpoint="$previous/interrupted.pt"; fi
checkpoint="${1:-$checkpoint}"
is_height="$("$python_bin" -c 'import sys,torch; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(int(bool((x.get("training_metadata") or {}).get("height_curriculum"))))' "$checkpoint")"
if [[ "$is_height" != "1" && -z "${1:-}" ]]; then
    checkpoint="runs/height_setup_20261012_160318/selected_best.pt"
fi
if [[ -z "${LANDING_TOTAL_STEPS:-}" ]]; then
    LANDING_TOTAL_STEPS="$("$python_bin" - "$checkpoint" "$previous" <<'PY'
import json,sys,torch
from pathlib import Path
x=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
steps=x['steps']
config=Path(sys.argv[2])/'config.json'
prior=json.loads(config.read_text()) if config.exists() else {}
old_budget=int(prior.get('arguments',{}).get('steps',0))
print(old_budget if (x.get('training_metadata') or {}).get('height_curriculum') and old_budget>steps else steps+180000)
PY
)"
    export LANDING_TOTAL_STEPS
fi
exec bash start_variable_motion_training.sh "$checkpoint"
