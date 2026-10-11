"""Smooth visual landing: own-command memory, bounded command jerk, soft contact.

Privileged simulator state is used ONLY for training labels/rewards/metrics.
Evaluation executes the network using images and its own state/command memory.
"""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pybullet as p
import torch

from .agent import DrQV2, Replay
from .config import AgentConfig
from .environment import LIMITS, MovingPlatformLanding
from .guided import Demonstrations, clone_step, stage_config

METHOD = 'Assisted visual DrQ-v2 + command memory + smoothness losses + bounded command jerk + soft-contact shaping'


def smooth_config(stage=1):
    return replace(stage_config(stage), smooth_observation=True, action_smooth_weight=.10,
                   soft_landing_weight=.10, acceleration_limit_xy=2., acceleration_limit_z=.8,
                   jerk_limit_xy=8., jerk_limit_z=4., velocity_response_seconds=.10)


def smooth_teacher(env):
    # TRAINING ONLY: tracking label with continuous slowdown near the surface.
    error = env.platform_position[:2]-env.state[:2]
    height = env.state[2]-env.cfg.platform_top
    x = np.clip((height-.10)/.70, 0., 1.)
    descent = .12 + .23*x*x*(3.-2.*x)
    if height > 1.5:
        descent = env.cfg.teacher_descent_fast
    vz = -descent*np.clip((.38-np.linalg.norm(error))/.22, 0., 1.)
    target = np.clip(np.r_[env.platform_velocity[:2]+2.*error, vz]/LIMITS, -1., 1.)
    return (env.previous_command+.25*(target-env.previous_command)).astype(np.float32)


def transfer(source):
    old, metadata = DrQV2.load(source)
    if old.cfg.proprio_dim == 30:
        return old, metadata
    if old.cfg.proprio_dim != 21:
        raise ValueError('Expected original 21D or smooth 30D checkpoint')
    new = DrQV2(replace(old.cfg, proprio_dim=30))
    for name in ('encoder', 'actor', 'critic', 'target'):
        original = getattr(old, name).state_dict()
        destination = getattr(new, name).state_dict()
        for key, value in original.items():
            if value.shape == destination[key].shape:
                destination[key] = value
            elif key == 'trunk.0.weight':
                expanded = torch.zeros_like(destination[key])
                expanded[:, :value.shape[1]] = value
                destination[key] = expanded
            else:
                raise ValueError((name, key, value.shape))
        getattr(new, name).load_state_dict(destination)
    # Changed input shape: fresh optimizers and a NEW update counter, not an exact resume.
    return new, metadata


class FlightLog:
    fields = ['time_s','x','y','z','platform_x','platform_y','platform_z',
              'relative_x','relative_y','relative_z','vx','vy','vz',
              'action_x','action_y','action_z','applied_vx','applied_vy','applied_vz',
              'command_ax','command_ay','command_az']

    def __init__(self, env):
        self.rows = []
        self.add(env, np.zeros(3), {})

    def add(self, env, action, info):
        self.rows.append([env.elapsed, *env.state[:3], *env.platform_position,
            *(env.state[:3]-env.platform_position-np.array([0.,0.,.04])), *env.state[10:13], *action,
            *env.applied_velocity, *env.command_acceleration])

    def save(self, path):
        with Path(path).open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.fields)
            writer.writerows(self.rows)

    def summary(self, info):
        rows = np.asarray(self.rows)
        # Exclude terminal sample from derivative metrics to omit contact impulses.
        flight = rows[:-1] if len(rows) > 3 else rows
        dt = np.diff(flight[:,0])
        accel = np.diff(flight[:,10:13], axis=0)/dt[:,None]
        jerk = np.diff(accel,axis=0)/((dt[1:]+dt[:-1])*.5)[:,None]
        changes = np.diff(flight[:,13:16],axis=0)
        contact = info.get('pre_contact_relative_velocity')
        return dict(event=info['event'], success=bool(info['success']),
            soft_landing=bool(info.get('soft_landing',False)), time_s=info['time_s'],
            xy_error_m=float(np.linalg.norm(info['relative_position'][:2])),
            action_delta_rms=float(np.sqrt(np.mean(np.sum(changes**2,axis=1)))),
            acceleration_rms_m_s2=float(np.sqrt(np.mean(np.sum(accel**2,axis=1)))),
            jerk_rms_m_s3=float(np.sqrt(np.mean(np.sum(jerk**2,axis=1)))) if len(jerk) else 0.,
            path_length_m=float(np.linalg.norm(np.diff(rows[:,1:4],axis=0),axis=1).sum()),
            contact_relative_velocity_m_s=contact,
            contact_xy_speed_m_s=None if contact is None else float(np.linalg.norm(contact[:2])),
            contact_z_speed_m_s=None if contact is None else float(abs(contact[2])))


def evaluate_flights(agent, config, output, episodes=5, seed=70001, panel=None, group='test'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = MovingPlatformLanding(config)
    rows = []
    updates = agent.updates
    try:
        for ep in range(episodes):
            if panel and panel.stop:
                break
            obs, _ = env.reset(seed=seed+ep)
            trajectory = FlightLog(env)
            ret = 0.
            while True:
                action = agent.act(obs,evaluation=True)
                obs, reward, term, trunc, info = env.step(action)
                trajectory.add(env,action,info)
                ret += reward
                if panel:
                    panel.validation_frame(env,group,ep+1,episodes,info,action,ret,term or trunc,
                                           getattr(panel,'training_step',0),getattr(panel,'training_total',0),updates)
                    panel.wait_if_paused()
                if term or trunc or (panel and panel.stop):
                    break
            trajectory.save(output/f'seed_{seed+ep}.csv')
            row = dict(seed=seed+ep, **trajectory.summary(info))
            rows.append(row)
            print('NETWORK TEST',group,row,flush=True)
    finally:
        env.close()
    assert agent.updates == updates, 'Evaluation must not update model weights'
    result = dict(method=METHOD if config.smooth_observation else 'Previous assisted visual policy',
                  episodes=len(rows), successes=sum(r['success'] for r in rows),
                  soft_landings=sum(r['soft_landing'] for r in rows), config=config.to_dict(),
                  seed_start=seed, exploration_noise=False, rows=rows)
    for metric in ('action_delta_rms','acceleration_rms_m_s2','jerk_rms_m_s3'):
        result[metric+'_mean'] = float(np.mean([r[metric] for r in rows])) if rows else None
    (output/'results.json').write_text(json.dumps(result,indent=2))
    return result


def prepare(args):
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    agent, original = transfer(args.checkpoint)
    data = Demonstrations(12000,args.seed,30)
    env = MovingPlatformLanding(smooth_config(0))
    collection = []
    try:
        for ep in range(args.episodes):
            stage = ep%3
            env.cfg = smooth_config(stage)
            obs, _ = env.reset(seed=args.seed*1000+ep)
            while True:
                action = smooth_teacher(env)
                data.add(obs,action)
                obs,reward,term,trunc,info = env.step(action)
                if term or trunc:
                    break
            collection.append(dict(episode=ep+1,stage=stage,event=info['event'],soft=info['soft_landing']))
            print('SMOOTH TEACHER DATA ONLY',collection[-1],'samples',data.size,flush=True)
    finally:
        env.close()
    data.save(out/'demonstrations.npz')
    (out/'teacher_collection.json').write_text(json.dumps(collection,indent=2))
    for update in range(args.bc_updates):
        metric = clone_step(agent,data.sample(32))
        if (update+1)%100 == 0:
            print('SMOOTH BC',update+1,metric,flush=True)
    agent.save(out/'initial.pt',smooth_config(1),0,dict(method=METHOD,
        source_checkpoint=str(args.checkpoint),source_steps=original.get('steps'),
        source_updates=original.get('updates'),bc_updates=args.bc_updates,stage=1,
        optimizer_reset=original['agent_config'].get('proprio_dim',21)!=30))


def benchmark(args):
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    baseline,_ = DrQV2.load(args.baseline)
    smooth,_ = DrQV2.load(args.checkpoint)
    results = {}
    for stage in args.stages:
        results[str(stage)] = {}
        for name, agent, config in [('baseline',baseline,stage_config(stage)),
                                    ('smooth',smooth,smooth_config(stage))]:
            results[str(stage)][name] = evaluate_flights(agent,config,out/f'stage{stage}_{name}',
                                                       args.episodes,args.seed)
    (out/'comparison.json').write_text(json.dumps(results,indent=2))
    plot_comparison(out,results)


def plot_comparison(out, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(121,projection='3d')
    rel = fig.add_subplot(122)
    stage = next(iter(results))
    for name,color in [('baseline','tab:orange'),('smooth','tab:blue')]:
        for index,row in enumerate(results[stage][name]['rows']):
            data = np.genfromtxt(out/f'stage{stage}_{name}'/f'seed_{row["seed"]}.csv',delimiter=',',names=True)
            label = name if index==0 else None
            ax.plot(data['x'],data['y'],data['z'],color=color,alpha=.65,label=label)
            rel.plot(np.hypot(data['relative_x'],data['relative_y']),data['relative_z'],color=color,alpha=.65,label=label)
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)',title=f'All test flights, stage {stage}')
    rel.set(xlabel='Distance to platform center (m)',ylabel='Height above deck (m)',title='Platform-relative trajectories')
    ax.legend(); rel.legend(); fig.tight_layout()
    fig.savefig(out/'trajectories.png',dpi=150)
    plt.close(fig)


def train(args):
    out = args.output
    out.mkdir(parents=True,exist_ok=True)
    agent,source = DrQV2.load(args.checkpoint)
    if agent.cfg.proprio_dim != 30:
        raise ValueError('Run smooth prepare first')
    data = Demonstrations.load(args.data,args.seed)
    online = Demonstrations(4000,args.seed+1,30)
    agent.cfg.noise_start,agent.cfg.noise_end,agent.cfg.noise_decay_steps = .03,.005,20000
    torch.manual_seed(args.seed)
    config = smooth_config(args.stage)
    env = MovingPlatformLanding(config,gui=args.gui)
    panel = None
    if args.gui:
        from .live_training import LiveTraining
        panel = LiveTraining(out)
        panel.root.title('SMOOTH VISUAL LANDING TRAINING | new model | not pure paper DrQ')
        panel.heading.configure(text='SMOOTH VISUAL LANDING - LEARNING IN PROGRESS')
        panel.focus = 'overview'
        panel.training_total = args.steps
    replay = Replay(4000,agent.cfg.n_step,agent.cfg.discount,args.seed)
    obs,_ = env.reset(seed=args.seed*10000)
    flight = FlightLog(env)
    trail_position = env.state[:3]
    start = time.monotonic()
    episode = successes = softs = completed = stable = 0
    ret = 0.
    metrics = {}
    validations = []
    best = (-1,-1,float('-inf'))
    reason = 'budget_reached'
    (out/'config.json').write_text(json.dumps(dict(method=METHOD,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        environment=config.to_dict(),source_checkpoint=str(args.checkpoint),
        stop_rule='3 consecutive fixed-stage 20-episode validations: success >=80%, soft >=60%, success-rate range <=5 percentage points',
        note='Operational stopping rule, not proof of convergence; best model preserved. Teacher labels train only.'),indent=2))
    def save(name):
        agent.save(out/name,config,completed,dict(method=METHOD,stage=args.stage,
            source_checkpoint=str(args.checkpoint),source_metadata=source.get('training_metadata'),
            stable_validations=stable,exact_resume=False))
    try:
        with (out/'episodes.jsonl').open('w') as log, (out/'metrics.jsonl').open('w') as metric_log:
            for step in range(args.steps):
                if panel:
                    panel.wait_if_paused()
                    if panel.stop:
                        reason='user_stopped'; break
                label = smooth_teacher(env)
                if env.visible:
                    online.add(obs,label)
                action = agent.act(obs,step)
                nxt,reward,term,trunc,info = env.step(action)
                flight.add(env,action,info)
                if panel and step%3==0:
                    p.addUserDebugLine(trail_position,env.state[:3],[.05,.25,1.],2,physicsClientId=env.client)
                    trail_position=env.state[:3]
                replay.add(obs,action,reward,nxt,term,trunc)
                obs=nxt; completed=step+1; ret+=reward
                if replay.size>=32 and step%4==0:
                    if online.size:
                        a,b=data.sample(16),online.sample(16)
                        expert={k:np.concatenate([a[k],b[k]]) for k in a}
                    else:
                        expert=data.sample(32)
                    metrics=agent.update(replay.sample(32),step,expert,bc_weight=3.,rl_weight=.02,
                                         gradient_clip=5.,smooth_weight=.15)
                    # Extra imitation update is deliberate and recorded in the method.
                    metrics.update(clone_step(agent,expert))
                    metric_log.write(json.dumps(dict(step=completed,**metrics))+'\n')
                if panel:
                    panel.training_step=completed
                    panel.training_label=f'SMOOTH MODEL TRAINING | Stage {args.stage} | Soft landings {softs}'
                    panel.saved_result.configure(text='30D own state + camera; smoothness losses + command jerk limits + soft-contact target.\nBlue 3D trail is the actual flight path. Noise is present during training; validation has no exploration noise.')
                    panel.update(env,completed,args.steps,0,agent.updates,episode,successes,
                        replay.size,ret,action,info,metrics,time.monotonic()-start,force=term or trunc)
                if term or trunc:
                    episode+=1; successes+=int(info['success']); softs+=int(info['soft_landing'])
                    row=dict(episode=episode,step=completed,**flight.summary(info))
                    flight.save(out/f'train_episode_{episode:04d}.csv')
                    log.write(json.dumps(row)+'\n'); log.flush()
                    print('SMOOTH NETWORK TRAIN',row,flush=True)
                    if panel:
                        panel.episode_result(episode,info,ret); panel.hold_result(.4)
                    obs,_=env.reset(seed=args.seed*10000+episode)
                    flight=FlightLog(env); ret=0.
                    trail_position=env.state[:3]
                if completed%1000==0:
                    save('latest.pt'); metric_log.flush()
                    print('SMOOTH TRAIN steps',completed,'updates',agent.updates,flush=True)
                if completed%args.eval_every==0:
                    score=evaluate_flights(agent,config,out/f'validation_{completed}',args.eval_episodes,80001,panel,'fixed-stage smooth test')
                    validations.append(dict(step=completed,**score))
                    (out/'validations.json').write_text(json.dumps(validations,indent=2))
                    key=(score['successes'],score['soft_landings'],-score['jerk_rms_m_s3_mean']) if score['episodes'] else (-1,-1,float('-inf'))
                    if key>best:
                        best=key; save('best.pt')
                    good=score['episodes']==args.eval_episodes and args.eval_episodes>=20 and score['successes']/score['episodes']>=.8 and score['soft_landings']/score['episodes']>=.6
                    stable=stable+1 if good else 0
                    if panel:
                        panel.heading.configure(text='SMOOTH VISUAL LANDING - LEARNING IN PROGRESS')
                    if stable>=3:
                        recent=[v['successes']/v['episodes'] for v in validations[-3:]]
                        if max(recent)-min(recent)<=.0500001:
                            reason='performance_target_stable'; break
            save('latest.pt')
            if not (out/'best.pt').exists():
                save('best.pt')
        final=dict(steps=completed,updates=agent.updates,episodes_completed=episode,
            training_successes=successes,soft_landings=softs,stop_reason=reason,
            wall_seconds=time.monotonic()-start,method=METHOD,status='stopped' if reason=='user_stopped' else 'completed')
        (out/'summary.json').write_text(json.dumps(final,indent=2))
        if panel and not panel.stop:
            panel.show_completed(final)
            panel.saved_result.configure(text=f'Saved best.pt and latest.pt. Stop reason: {reason}. Click test to watch this 30D policy without learning.')
            while not panel.stop:
                panel.pump()
                if panel.replay_requested:
                    panel.replay_requested=False
                    panel.begin_review()
                    evaluate_flights(agent,config,out/'desktop_test',5,90001,panel,'saved smooth model')
                    panel.review_finished()
                time.sleep(.05)
    except BaseException as exc:
        save('interrupted.pt')
        (out/'failure.json').write_text(json.dumps(dict(step=completed,reason=repr(exc))))
        raise
    finally:
        env.close()
        if panel:
            panel.close()


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','benchmark','train'):
        q=sub.add_parser(name)
        q.add_argument('--checkpoint',type=Path,required=True)
        q.add_argument('--output',type=Path,required=True)
        q.add_argument('--seed',type=int,default=31 if name!='benchmark' else 70001)
        if name=='prepare':
            q.add_argument('--episodes',type=int,default=24)
            q.add_argument('--bc-updates',type=int,default=1200)
        elif name=='benchmark':
            q.add_argument('--baseline',type=Path,required=True)
            q.add_argument('--episodes',type=int,default=5)
            q.add_argument('--stages',type=int,nargs='+',default=[1,2])
        else:
            q.add_argument('--data',type=Path,required=True)
            q.add_argument('--stage',type=int,default=1)
            q.add_argument('--steps',type=int,default=60000)
            q.add_argument('--eval-every',type=int,default=5000)
            q.add_argument('--eval-episodes',type=int,default=20)
            q.add_argument('--gui',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare': prepare(args)
    elif args.command=='benchmark': benchmark(args)
    else: train(args)


if __name__=='__main__':
    main()
