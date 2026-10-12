"""Safely checkpoint this repository's trainer before WSLg display recovery."""
import argparse,json,os,re,signal,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

def inspect():
    run=Path((ROOT/'runs/current_smooth_training.txt').read_text().strip())
    if not run.is_absolute(): run=ROOT/run
    run=run.resolve();run.relative_to(ROOT/'runs')
    pid=int((run/'training.pid').read_text())
    try: command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError: command=[]
    alive=b'landing_repro.smooth' in command and str(run.relative_to(ROOT)).encode() in command
    probe=subprocess.run(['xwininfo','-root','-tree'],capture_output=True,text=True)
    names=re.findall(r'^\s*0x[0-9a-f]+ "([^"]+)"',probe.stdout,re.M)
    allowed=('UAV LANDING TRAINING','Bullet Physics ExampleBrowser','SMOOTH VISUAL LANDING TRAINING',
             'LIVE DrQ-v2 TRAINING','ASSISTED VISUAL MODEL TRAINING','Saved DrQ-v2 model test')
    other=[name for name in names if name!='Weston WM' and not name.startswith(allowed)]
    return dict(run=str(run),pid=pid,alive=alive,other_gui_windows=other)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['inspect','save']);args=parser.parse_args()
    state=inspect()
    if args.command=='save':
        if state['other_gui_windows']:
            raise RuntimeError('Other Linux GUI windows are open; display recovery stopped: '+repr(state['other_gui_windows']))
        if state['alive']:
            os.kill(state['pid'],signal.SIGINT)
            deadline=time.monotonic()+30
            while Path(f"/proc/{state['pid']}/cmdline").exists():
                try: command=Path(f"/proc/{state['pid']}/cmdline").read_bytes()
                except FileNotFoundError: break
                if b'landing_repro.smooth' not in command: break
                if time.monotonic()>deadline: raise TimeoutError('Trainer has not finished saving; display service was not touched')
                time.sleep(.2)
        run=Path(state['run']); candidates=[p for p in (run/'latest.pt',run/'interrupted.pt') if p.exists()]
        if not candidates: raise RuntimeError('No saved checkpoint; display service was not touched')
        import torch
        checkpoint=max(candidates,key=lambda p:p.stat().st_mtime_ns)
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if not (data.get('training_metadata') or {}).get('height_curriculum'):
            raise RuntimeError('This recovery entry supports the current height curriculum only')
        state.update(checkpoint=str(checkpoint),saved_steps=data['steps'],verified=True)
        (run/'display_recovery_checkpoint.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state))

if __name__=='__main__': main()
