"""Frozen development regression: old near mode versus faster bounded finish."""
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
    args=parser.parse_args()
    checksum=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    agent,metadata=DrQV2.load(args.checkpoint)
    updates=agent.updates
    result={}
    for name,enabled in [('previous_near',False),('terminal_finish',True)]:
        rows=[]
        for seed in (70001,410064):
            test=evaluate_flights(agent,smooth_config(1,True,True,enabled),
                args.output/name/f'seeds_{seed}',3,seed,group='FROZEN / NO LEARNING '+name)
            rows.extend(test['rows'])
        result[name]=dict(episodes=len(rows),successes=sum(row['success'] for row in rows),
            soft_landings=sum(row['soft_landing'] for row in rows),rows=rows,
            mean_time_s=sum(row['time_s'] for row in rows)/len(rows),
            mean_jerk_rms_m_s3=sum(row['jerk_rms_m_s3'] for row in rows)/len(rows))
    assert updates==agent.updates and checksum==hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report=dict(checkpoint=str(args.checkpoint),checkpoint_steps=metadata['steps'],sha256=checksum,
        optimization_updates=0,results=result,
        note='Development regression only: includes seeds associated with reported failures, frozen weights. Not an exact replay of episode 66 or a held-out statistical result.')
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({name:{key:value for key,value in data.items() if key!='rows'} for name,data in result.items()},indent=2))


if __name__=='__main__': main()
