"""Independent paper-derived experiment runner; never loads assisted checkpoints."""
import argparse,json,os,signal,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from .protocols import LADOSZ_PAPER,ladosz_configs
from .agent import PaperAgent
from .environment import PaperLanding
from .replay import DiskReplay

METHOD='Ladosz2024 independent DrQ-v2 / Fig3 BatchNorm / no expert or prediction assistance'

def write_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def evaluate(agent,cfg,out,episodes=10,seed_start=800001,constant_descent=False,panel=None):
    env=PaperLanding(cfg,constant_descent=constant_descent);rows=[];updates=agent.updates
    try:
        for ep in range(episodes):
            obs,_=env.reset(seed=seed_start+ep);ret=0.
            while True:
                if panel:
                    panel.pump()
                    if panel.stop:break
                obs,reward,term,trunc,info=env.step(agent.act(obs,evaluation=True));ret+=reward
                if term or trunc:break
            if panel and panel.stop:break
            rows.append(dict(seed=seed_start+ep,event=info['event'],success=info['success'],
                time_s=info['time_s'],xy_error_m=float(np.linalg.norm(info['relative_position'][:2])),episode_return=ret))
    finally:env.close()
    assert agent.updates==updates
    result=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
        mean_return=float(np.mean([r['episode_return'] for r in rows])) if rows else None,
        seeds=[r['seed'] for r in rows],optimization_updates=0,rows=rows)
    write_json(out,result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--episodes',type=int,default=4000);p.add_argument('--max-steps',type=int,default=3600000)
    p.add_argument('--variant',choices=['baseline','no_los','constant_descent'],default='baseline')
    p.add_argument('--reward-interpretation',choices=['normalized','literal'],default='normalized')
    p.add_argument('--vertical-reward',choices=['literal','actual_guard'],default='actual_guard')
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    p.add_argument('--gui',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--eval-every-episodes',type=int,default=100);p.add_argument('--final-eval-episodes',type=int,default=500)
    p.add_argument('--checkpoint-every',type=int,default=1000)
    a=p.parse_args()
    if a.seed not in range(1,6):p.error('Paper seeds are 1..5')
    cfg,acfg=ladosz_configs(a.seed,a.reward_interpretation,a.variant=='no_los',a.vertical_reward)
    if a.resume:
        spec=json.loads((a.output/'config.json').read_text())
        expected=dict(spec['arguments']);current={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        for key in ('seed','variant','reward_interpretation','vertical_reward'):
            if expected[key]!=current[key]:raise ValueError('Resume task mismatch: '+key)
        agent,source=PaperAgent.load(a.output/'latest.pt',device=a.device)
        if (source.get('training_metadata') or {}).get('method')!=METHOD:raise ValueError('Assisted checkpoint rejected')
        meta=source['training_metadata'];completed=source['steps'];episodes=meta['episodes'];successes=meta['successes'];best=meta['best_mean_return']
        rng=np.random.default_rng();rng.bit_generator.state=meta['numpy_rng'];torch.set_rng_state(meta['torch_rng'])
        if agent.device.type=='cuda' and meta.get('torch_cuda_rng') is not None:
            states=meta['torch_cuda_rng']
            if len(states)!=torch.cuda.device_count():raise ValueError('CUDA RNG device count changed; resume with the same visible GPU count')
            torch.cuda.set_rng_state_all(states)
        replay=DiskReplay(a.output/'replay',acfg.replay_capacity,acfg.n_step,acfg.discount,a.seed,resume=True)
    else:
        if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Use a new run or --resume')
        a.output.mkdir(parents=True,exist_ok=True)
        agent=PaperAgent(acfg,device=a.device);completed=episodes=successes=0;best=None;rng=np.random.default_rng(a.seed)
        replay=DiskReplay(a.output/'replay',acfg.replay_capacity,acfg.n_step,acfg.discount,a.seed)
    spec=dict(method=METHOD,claim='Independent paper-derived implementation; author ambiguities remain. No numerical reproduction claimed.',
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},environment=cfg.to_dict(),agent=acfg.to_dict(),
        published=LADOSZ_PAPER,
        choices={'own_state':'3 x normalized world velocity and xyzw quaternion; encoding not fully specified in paper',
          'clock':'300Hz physics / 60Hz PID / 30Hz policy; 10 physics ticks per policy action',
          'entropy_temperature':'not implemented: unresolved conflict with deterministic DrQ-v2',
          'update_every':2,'noise_schedule':'linear 1.0 to 0.1 over 50k steps; adapter choice',
          'vertical_reward':a.vertical_reward,'reward_exponent':a.reward_interpretation,'terminal_los_gate':'outer Eq7 gate retained',
          'network':'Fig3 BatchNorm; sigmoid actor transformed to [-1,1]',
          'target_bn_statistics':'Polyak average of floating buffers; copy integer counters',
          'camera':'90 degree FOV; synthetic H pad; exact original calibration/photo unavailable',
          'resume':'network, optimizer, replay and RNG restored; physics and pending n-step tail reset',
          'checkpoint_atomicity':'single writer; crash between replay and checkpoint save may require prior checkpoint',
          'baseline_scope':'no image/domain randomization; separate missing-assets protocol describes randomized variant',
          'training_device':str(agent.device),'device_resume':'Cross-device resume preserves weights/optimizers, not an identical random trajectory'})
    write_json(a.output/'config.json',spec);(a.output/'training.pid').write_text(str(os.getpid()))
    panel=None;env=None;pool=ThreadPoolExecutor(max_workers=1);stop=[False];metrics={};started=time.monotonic();reason='budget_reached';last_event='running'
    def request_stop(*_):stop[0]=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    def save(name='latest.pt'):
        replay.flush()
        agent.save(a.output/name,cfg,completed,dict(method=METHOD,episodes=episodes,successes=successes,
            best_mean_return=best,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            torch_cuda_rng=torch.cuda.get_rng_state_all() if agent.device.type=='cuda' else None,
            training_device=str(agent.device),
            reward_interpretation=a.reward_interpretation,vertical_reward=a.vertical_reward,variant=a.variant,exact_physics_resume=False))
    def status(state):
        return dict(state=state,device=str(agent.device),method=METHOD,seed=a.seed,step=completed,updates=agent.updates,episodes=episodes,
            successes=successes,replay_size=replay.size,warmup=completed<acfg.warmup_steps,
            last_event=last_event,critic_loss=metrics.get('critic_loss'),wall_seconds=time.monotonic()-started,
            initial_height_m=cfg.initial_height,variant=a.variant)
    try:
        env=PaperLanding(cfg,gui=a.gui,constant_descent=a.variant=='constant_descent')
        if a.gui:
            from .monitor import Monitor
            panel=Monitor(a.output)
        obs,_=env.reset(seed=a.seed*100000+episodes);ret=0.;save()
        while completed<a.max_steps and episodes<a.episodes:
            if panel:
                panel.wait()
                if panel.stop:stop[0]=True
            if stop[0]:reason='stopped_and_saved';break
            action=rng.uniform(-1,1,3).astype(np.float32) if completed<acfg.warmup_steps else agent.act(obs,completed)
            next_obs,reward,term,trunc,info=env.step(action)
            if not np.isfinite(reward):raise FloatingPointError('Nonfinite physical reward')
            replay.add(obs,action,reward,next_obs,term,trunc);obs=next_obs;ret+=reward;completed+=1
            if completed>=acfg.warmup_steps and replay.size>=acfg.batch_size and completed%acfg.update_every==0:
                if panel:panel.update(env,status('gradient_update'),force=True)
                future=pool.submit(agent.update,replay.sample(acfg.batch_size),completed)
                while not future.done():
                    if panel:panel.pump()
                    time.sleep(.05)
                metrics=future.result()
                with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=completed,**metrics))+'\n')
            if panel:panel.update(env,status('training'))
            elif completed%100==0:write_json(a.output/'live_status.json',status('training'))
            if term or trunc:
                episodes+=1;successes+=int(info['success']);last_event=info['event']
                row=dict(episode=episodes,step=completed,event=last_event,success=info['success'],episode_return=ret,time_s=info['time_s'])
                with (a.output/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print('PAPER_EPISODE',json.dumps(row),flush=True)
                if panel:panel.message(f"Episode {episodes}: {last_event} | return {ret:.2f}")
                if a.eval_every_episodes>0 and episodes%a.eval_every_episodes==0:
                    save();write_json(a.output/'live_status.json',status('validating'))
                    result=evaluate(agent,cfg,a.output/f'validation_{episodes:04d}.json',10,800001+a.seed*1000,
                                    a.variant=='constant_descent',panel)
                    if result['episodes']==10 and (best is None or result['mean_return']>best):
                        best=result['mean_return'];save('best.pt')
                obs,_=env.reset(seed=a.seed*100000+episodes);ret=0.
            if completed%a.checkpoint_every==0:save();print('PAPER_STEP',completed,agent.updates,flush=True)
        save();write_json(a.output/'live_status.json',status(reason))
        if episodes>=a.episodes and a.final_eval_episodes and not stop[0] and not(panel and panel.stop):
            selected,_=PaperAgent.load(a.output/'best.pt' if (a.output/'best.pt').exists() else a.output/'latest.pt',device=a.device)
            evaluate(selected,cfg,a.output/'final_evaluation.json',a.final_eval_episodes,900001+a.seed*1000,
                     a.variant=='constant_descent',panel)
        write_json(a.output/'summary.json',dict(status(reason),numerical_reproduction_complete=False))
    except BaseException as exc:
        try:save()
        except Exception:pass
        write_json(a.output/'failure.json',dict(step=completed,reason=repr(exc),method=METHOD));raise
    finally:
        pool.shutdown(wait=True)
        if env:env.close()
        if panel:panel.close()
if __name__=='__main__':main()