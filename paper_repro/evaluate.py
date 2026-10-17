"""Frozen checkpoint evaluation, using saved environment and distinct evaluation seeds."""
import argparse,hashlib,time
from dataclasses import asdict
from pathlib import Path
from .agent import PaperAgent
from .environment import PaperLanding
from .train import METHOD,write_json
from landing_repro.config import EnvConfig
import numpy as np
from PIL import Image


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--seed-start',type=int,default=901001)
    p.add_argument('--gui',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    args=p.parse_args()
    if args.episodes<1:p.error('episodes must be positive')
    if args.output.exists():p.error('Choose a new output directory; existing evaluation is preserved')
    before_hash=sha256(args.checkpoint)
    agent,data=PaperAgent.load(args.checkpoint,device=args.device)
    meta=data.get('training_metadata') or {}
    if meta.get('method')!=METHOD:raise ValueError('Use a paper_repro checkpoint, not an assisted policy')
    cfg=EnvConfig(**data['env_config']);args.output.mkdir(parents=True)
    env=PaperLanding(cfg,gui=args.gui,constant_descent=meta.get('variant')=='constant_descent')
    panel=None;rows=[];started=time.monotonic();initial_updates=agent.updates
    if args.gui:
        from .monitor import Monitor
        panel=Monitor(args.output,evaluation=True)
    try:
        for episode in range(args.episodes):
            seed=args.seed_start+episode;obs,info=env.reset(seed=seed);total=0.;visibility=steps=0;end_frame=None;actions=[]
            initial=dict(position=env.state[:3].tolist(),platform_velocity=env.platform_velocity.tolist())
            if episode==0:Image.fromarray(env.render()).resize((504,504),Image.Resampling.NEAREST).save(args.output/'onboard_initial.png')
            while True:
                if panel:
                    panel.wait()
                    if panel.stop:break
                action=agent.act(obs,evaluation=True);actions.append(action.copy())
                obs,reward,term,trunc,info=env.step(action)
                total+=reward;steps+=1;visibility+=int(info['visible']);end_frame=env.render()
                if panel:panel.update(env,dict(state='frozen_evaluation',step=steps,updates=agent.updates,
                    warmup=False,replay_size=0,episodes=episode,successes=sum(r['success'] for r in rows),seed=seed,critic_loss=None,
                    wall_seconds=time.monotonic()-started))
                if term or trunc:break
            if panel and panel.stop:break
            Image.fromarray(end_frame).resize((504,504),Image.Resampling.NEAREST).save(args.output/f'onboard_final_{episode+1:02d}.png')
            rows.append(dict(seed=seed,initial=initial,event=info['event'],success=info['success'],
                time_s=info['time_s'],steps=steps,episode_return=total,
                xy_error_m=float(np.linalg.norm(info['relative_position'][:2])),
                visible_fraction=visibility/steps,
                normalized_action_mean=np.mean(actions,axis=0).tolist(),
                saturated_action_component_fraction=float(np.mean(np.abs(actions)>.95)),
                pre_contact_relative_velocity=info['pre_contact_relative_velocity']))
            print('FROZEN_EVALUATION',rows[-1],flush=True)
    finally:
        env.close()
        if panel:panel.close()
    assert agent.updates==initial_updates
    assert before_hash==sha256(args.checkpoint),'Checkpoint file changed during evaluation'
    result=dict(method=METHOD,device=str(agent.device),checkpoint=str(args.checkpoint),checkpoint_sha256=before_hash,
        checkpoint_training_steps=data['steps'],checkpoint_training_updates=data['updates'],
        environment=asdict(cfg),variant=meta.get('variant'),episodes=len(rows),
        successes=sum(r['success'] for r in rows),
        success_rate=sum(r['success'] for r in rows)/len(rows) if rows else None,
        optimization_updates=0,checkpoint_unchanged=True,wall_seconds=time.monotonic()-started,rows=rows,
        claim='Small frozen diagnostic; not the paper final 500-episode evaluation.')
    write_json(args.output/'evaluation.json',result)
    print('REPORT',args.output/'evaluation.json',flush=True)
if __name__=='__main__':main()
