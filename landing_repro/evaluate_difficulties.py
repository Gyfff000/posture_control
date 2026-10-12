"""Freeze one checkpoint and assess all four motion difficulty configurations."""
import argparse,hashlib,json
from pathlib import Path
import torch
from .agent import DrQV2
from .motion_curriculum import motion_config
from .smooth import evaluate_flights,smooth_config

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--episodes',type=int,default=20)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(); agent,metadata=DrQV2.load(args.checkpoint)
    torch.set_num_threads(1); updates=agent.updates; results=[]
    for phase in range(4):
        config=motion_config(smooth_config,phase,True,True,True)
        result=evaluate_flights(agent,config,args.output/f'phase_{phase}',args.episodes,80001+phase*1000,
            group=f'FROZEN ALL DIFFICULTIES phase {phase}')
        results.append(dict(phase=phase,**result))
        (args.output/'progress.json').write_text(json.dumps(dict(checkpoint_steps=metadata['steps'],results=results),indent=2))
    assert updates==agent.updates and sha==hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report=dict(checkpoint=str(args.checkpoint),checkpoint_steps=metadata['steps'],sha256=sha,optimization_updates=0,
        episodes_per_phase=args.episodes,results=results,
        note='Same selected checkpoint, frozen mean actions. Fixed validation seeds, not a statistically independent test set. Phase 3 is a transfer test without prior phase-3 training and has a higher initial height.')
    (args.output/'all_difficulties.json').write_text(json.dumps(report,indent=2))
    print('ALL_DIFFICULTIES_DONE',json.dumps([dict(phase=r['phase'],successes=r['successes'],soft=r['soft_landings'],episodes=r['episodes']) for r in results]),flush=True)
if __name__=='__main__': main()