"""Paired frozen-policy development regression for near-field controller v2."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from .agent import DrQV2
from .config import EnvConfig
from .motion_curriculum import motion_config
from .smooth import evaluate_flights,smooth_config


def baseline_module(name,path):
    spec=importlib.util.spec_from_file_location('landing_repro._baseline_'+name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    checksum=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    agent,metadata=DrQV2.load(args.checkpoint); updates=agent.updates
    old_center=baseline_module('center',args.baseline/'center_marker.py')
    old_guard=baseline_module('guard',args.baseline/'near_landing.py')
    configs={'before':EnvConfig(**metadata['env_config']),
             'after':motion_config(smooth_config,1,True,True,True)}
    results={}
    for name,config in configs.items():
        rows=[]
        with patch('landing_repro.center_marker.CenterTracker',old_center.CenterTracker) if name=='before' else nullcontext():
            with patch('landing_repro.near_landing.NearLandingGuard',old_guard.NearLandingGuard) if name=='before' else nullcontext():
                for seed in (410268,90001):
                    result=evaluate_flights(agent,config,args.output/name/f'seeds_{seed}',3,seed,
                        group='FROZEN / NO LEARNING '+name)
                    rows.extend(result['rows'])
        results[name]=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
            soft_landings=sum(r['soft_landing'] for r in rows),
            near_estimate_lost=sum(r['event']=='near_estimate_lost' for r in rows),
            partial_flow_frames=sum(r['partial_flow_frames'] for r in rows),rows=rows)
    assert updates==agent.updates and checksum==hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report=dict(checkpoint=str(args.checkpoint),checkpoint_steps=metadata['steps'],sha256=checksum,
        optimization_updates=0,results=results,
        note='Paired six-seed development check, not an exact replay of noisy training failures or a held-out statistical success rate. Includes failure-associated seeds. Actual physical contact defines success.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in results.items()},indent=2))


if __name__=='__main__': main()
