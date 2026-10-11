"""Frozen-checkpoint development trials; never calls any optimizer or trainer."""
import argparse
import hashlib
import json
from pathlib import Path

from .agent import DrQV2
from .smooth import evaluate_flights,smooth_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--seed',type=int,default=70001)
    args=parser.parse_args()
    original_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    agent,_=DrQV2.load(args.checkpoint)
    before=agent.updates
    results={}
    for name,enabled in [('prediction_only',False),('near_assist',True)]:
        results[name]=evaluate_flights(agent,smooth_config(1,True,enabled),
            args.output/name,args.episodes,args.seed,group='FROZEN CHECK / NO TRAINING '+name)
    assert agent.updates==before
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==original_hash
    report=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=original_hash,
        optimization_updates_performed=0,seed_start=args.seed,
        note='Small development check of existing frozen weights under changed controllers; not new model training or a statistical claim.',
        results=results)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({key:dict(successes=value['successes'],episodes=value['episodes'],
        soft_landings=value['soft_landings'],jerk=value['jerk_rms_m_s3_mean']) for key,value in results.items()},indent=2))


if __name__=='__main__': main()
