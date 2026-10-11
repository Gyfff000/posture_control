"""Linux/WSL: save the active smooth trainer and reopen with prediction guidance.

Weights/optimizers and cumulative step counters survive; replay, RNG and partial
flight reset. Run in the foreground so the desktop GUI remains attached.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch


def main():
    pointer=Path('runs/current_smooth_training.txt')
    old=Path(pointer.read_text().strip())
    pid=int((old/'training.pid').read_text())
    command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if b'landing_repro.smooth' not in command or b'train' not in command or str(old).encode() not in command:
        raise RuntimeError('PID is not the expected active trainer; nothing interrupted')
    config=json.loads((old/'config.json').read_text())
    os.kill(pid,signal.SIGINT)
    deadline=time.monotonic()+30
    while Path(f'/proc/{pid}/stat').exists():
        if Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].startswith('Z'):
            break
        if time.monotonic()>deadline:
            raise RuntimeError('Trainer has not exited; no second trainer started')
        time.sleep(.1)
    checkpoint=old/'interrupted.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    offset=saved['steps']
    args=config['arguments']
    if not 0<offset<int(args['steps']):
        raise RuntimeError('No remaining step budget; saved checkpoint left intact')
    out=Path('runs')/time.strftime('smooth_prediction_%Y%m%d_%H%M%S')
    out.mkdir()
    transition=dict(reason='Partial/lost camera detection: per-policy-step prediction, time-decayed confidence and prediction reward',
        previous_run=str(old),checkpoint=str(checkpoint),restored_total_steps=offset,
        weights_and_optimizers_preserved=True,exact_resume=False,
        note='Partial flight/replay/online buffer/RNG reset. Changed-task validation ranking starts fresh; previous best retained separately.')
    (out/'resume_transition.json').write_text(json.dumps(transition,indent=2))
    (old/'prediction_transition.json').write_text(json.dumps(transition,indent=2))
    (out/'training.pid').write_text(str(os.getpid()))
    pointer.write_text(str(out)+'\n')
    print(json.dumps(transition,indent=2),flush=True)
    # Keep the exec session alive; also retain full output if its UI truncates.
    log=os.open(out/'training.log',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o644)
    os.dup2(log,1); os.dup2(log,2); os.close(log)
    for key,value in dict(LIBGL_ALWAYS_SOFTWARE='1',LP_NUM_THREADS='2',OMP_NUM_THREADS='2',
                          MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1').items():
        os.environ[key]=value
    os.execv(sys.executable,[sys.executable,'-m','landing_repro.smooth','train',
        '--checkpoint',str(checkpoint),'--data',args['data'],'--output',str(out),
        '--steps',str(args['steps']),'--stage',str(args['stage']),'--seed',str(args['seed']),
        '--eval-every',str(args['eval_every']),'--eval-episodes',str(args['eval_episodes']),
        '--step-offset',str(offset),'--resume-run',str(old),'--prediction-guidance','--gui'])


if __name__=='__main__': main()
