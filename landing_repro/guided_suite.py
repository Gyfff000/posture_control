"""Two long fine-tuning runs, sequentially, with a live desktop GUI."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22])
    p.add_argument('--gui', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    progress = dict(method='Two fine-tuning runs sharing one saved assisted-visual initialization',
                    source_checkpoint=str(args.checkpoint), source_data=str(args.data),
                    steps_per_run=args.steps, seeds=args.seeds, runs=[], state='running')
    progress_path = args.output/'suite_status.json'
    progress_path.write_text(json.dumps(progress, indent=2))
    for i, seed in enumerate(args.seeds):
        out = args.output/f'seed_{seed}'
        cmd = [sys.executable, '-m', 'landing_repro.train_guided', '--checkpoint', str(args.checkpoint),
               '--data', str(args.data), '--output', str(out), '--steps', str(args.steps), '--seed', str(seed)]
        if args.gui:
            cmd.append('--gui')
        if i < len(args.seeds)-1:
            cmd.append('--auto-next')
        progress.update(active_seed=seed, active_output=str(out))
        progress_path.write_text(json.dumps(progress, indent=2))
        print(f'START RUN {i+1}/{len(args.seeds)} seed={seed} steps={args.steps}', flush=True)
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        progress['active_pid'] = child.pid
        progress_path.write_text(json.dumps(progress, indent=2))
        with (args.output/f'seed_{seed}_console.log').open('w') as log:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
        code = child.wait()
        summary_path = out/'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {'status':'failed'}
        progress['runs'].append(dict(seed=seed, return_code=code, summary=summary))
        progress_path.write_text(json.dumps(progress, indent=2))
        if code or summary['status'] == 'stopped':
            progress['state'] = 'failed' if code else 'stopped_by_user'
            progress_path.write_text(json.dumps(progress, indent=2))
            return
    progress['state'] = 'completed'
    progress_path.write_text(json.dumps(progress, indent=2))


if __name__ == '__main__':
    main()
