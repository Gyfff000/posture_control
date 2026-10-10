"""Long, demonstration-assisted visual learning with gated curriculum and evaluations."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .agent import DrQV2, Replay
from .config import EnvConfig
from .environment import MovingPlatformLanding
from .guided import METHOD, Demonstrations, clone_step, evaluate, stage_config, teacher_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=60000)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--stage', type=int, help='Default: stage saved in the checkpoint, or stage 1 for pretraining')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--auto-next', action='store_true', help='Keep end results for ten seconds, then let the suite start its next run')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    agent, metadata = DrQV2.load(args.checkpoint)
    demos = Demonstrations.load(args.data, args.seed)
    online = Demonstrations(8000, args.seed+1)
    rng = np.random.default_rng(args.seed)
    cfg = agent.cfg
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    cfg.noise_start, cfg.noise_end, cfg.noise_decay_steps = .08, .02, 40000
    source_training = metadata.get('training_metadata') or {}
    stage = args.stage if args.stage is not None else source_training.get('stage', 1)
    ecfg = stage_config(stage)
    env = MovingPlatformLanding(ecfg, gui=args.gui)
    panel = None
    if args.gui:
        from .live_training import LiveTraining
        panel = LiveTraining(args.output)
        panel.root.title('ASSISTED VISUAL MODEL TRAINING | curriculum | not pure paper DrQ')
        panel.heading.configure(text='VISUAL NETWORK TRAINING - demonstrations + DrQ-v2')
    replay = Replay(5000, cfg.n_step, cfg.discount, args.seed)
    (args.output/'config.json').write_text(json.dumps(dict(method=METHOD, seed=args.seed,
        requested_steps=args.steps, initial_stage=stage, checkpoint=str(args.checkpoint),
        source_steps=metadata.get('steps', 0),
        source_demonstrations=str(args.data), bc_weight=5., actor_rl_weight=.02,
        note='Training uses privileged action labels; evaluations do not use the teacher.'), indent=2))
    bc_updates = source_training.get('bc_updates', 0)
    start = time.monotonic()
    episode = successes = completed_steps = 0
    episode_return = 0.
    metrics = {}
    scores = []
    best_rate = -1.
    obs, _ = env.reset(seed=args.seed*10000)

    def save(name):
        agent.save(args.output/name, ecfg, completed_steps,
                   dict(method=METHOD, bc_updates=bc_updates, stage=stage, seed=args.seed,
                        training_steps=completed_steps, source_steps=metadata.get('steps', 0),
                        lineage_steps=metadata.get('steps', 0)+completed_steps,
                        source_checkpoint=str(args.checkpoint), exact_resume=False))

    try:
        with (args.output/'metrics.jsonl').open('w') as mf, (args.output/'episodes.csv').open('w', newline='') as ef:
            writer = csv.writer(ef)
            writer.writerow(['episode', 'stage', 'event', 'return', 'time_s', 'xy_error'])
            for step in range(args.steps):
                tick = time.monotonic()
                if panel:
                    panel.wait_if_paused()
                    if panel.stop:
                        break
                # Privileged labels supervise TRAINING, but are NEVER executed as flight actions here.
                label = teacher_action(env)
                if env.visible:
                    online.add(obs, label)
                action = agent.act(obs, step)
                next_obs, reward, term, trunc, info = env.step(action)
                replay.add(obs, action, reward, next_obs, term, trunc)
                obs = next_obs
                completed_steps = step+1
                episode_return += reward
                if replay.size >= cfg.batch_size and step % 4 == 0:
                    if online.size:
                        half = cfg.batch_size//2
                        left, right = demos.sample(half), online.sample(cfg.batch_size-half)
                        guidance = {k:np.concatenate([left[k], right[k]]) for k in left}
                    else:
                        guidance = demos.sample(cfg.batch_size)
                    metrics = agent.update(replay.sample(cfg.batch_size), step, expert=guidance,
                                           bc_weight=5., rl_weight=.02, gradient_clip=5.)
                    metrics.update(clone_step(agent, guidance))
                    bc_updates += 1
                    mf.write(json.dumps(dict(step=step+1, stage=stage, bc_updates=bc_updates, **metrics))+'\n')
                if panel:
                    panel.training_label = f'ASSISTED VISUAL TRAINING | curriculum stage {stage} | BC updates {bc_updates}'
                    panel.saved_result.configure(text=f'Run seed {args.seed} | Stage {stage}: height {ecfg.initial_height} +/- {ecfg.offset_z} m, platform 0-{ecfg.platform_speed_max} m/s.\n'
                        'Network actions + exploration noise. Teacher labels are used for training losses only.')
                    panel.update(env, completed_steps, args.steps, 0, agent.updates, episode, successes,
                                 replay.size, episode_return, action, info, metrics, time.monotonic()-start,
                                 force=term or trunc)
                if term or trunc:
                    episode += 1
                    successes += int(info['success'])
                    writer.writerow([episode, stage, info['event'], episode_return, info['time_s'], np.linalg.norm(info['relative_position'][:2])])
                    ef.flush()
                    print(f'NETWORK TRAIN episode={episode} stage={stage} event={info["event"]} successes={successes} steps={completed_steps}', flush=True)
                    if panel:
                        panel.episode_result(episode, info, episode_return)
                        panel.hold_result(.5)
                    obs, _ = env.reset(seed=args.seed*10000+episode)
                    episode_return = 0.
                if completed_steps % 1000 == 0:
                    save('latest.pt')
                    mf.flush()
                    print(f'ASSISTED step={completed_steps}/{args.steps} stage={stage} RL_updates={agent.updates} BC_updates={bc_updates}', flush=True)
                if completed_steps % args.eval_every == 0:
                    if panel:
                        panel.heading.configure(text='INDEPENDENT VALIDATION - training paused while model is tested')
                        panel.pump()
                    progress = panel.pump if panel else None
                    def observer(group):
                        if not panel:
                            return None
                        return lambda test_env, ep, count, inf, act, ret, done: panel.validation_frame(
                            test_env, group, ep, count, inf, act, ret, done,
                            completed_steps, args.steps, agent.updates)
                    current = evaluate(agent, stage_config(stage), 5, 30001, progress=progress,
                                       observer=observer('1/3: current difficulty'))
                    next_stage = evaluate(agent, stage_config(min(stage+1, 4)), 5, 31001, progress=progress,
                                          observer=observer('2/3: next difficulty'))
                    paper = evaluate(agent, EnvConfig(), 5, 32001, progress=progress,
                                     observer=observer('3/3: original task'))
                    entry = dict(step=completed_steps, stage=stage, current=current, next=next_stage, original_task=paper)
                    scores.append(entry)
                    (args.output/'validations.json').write_text(json.dumps(scores, indent=2))
                    print('VALIDATION', {k:f'{v["successes"]}/{v["episodes"]}' for k,v in entry.items() if isinstance(v,dict)}, flush=True)
                    if current['success_rate'] > best_rate:
                        best_rate = current['success_rate']
                        save(f'best_stage{stage}.pt')
                    if current['success_rate'] >= .8 and stage < 4:
                        stage += 1
                        ecfg = stage_config(stage)
                        env.cfg = ecfg
                        best_rate = -1.
                        # Record any interrupted partial episode as a curriculum boundary.
                        writer.writerow([episode+1, stage-1, 'curriculum_reset_partial', episode_return, env.elapsed, ''])
                        ef.flush()
                        obs, _ = env.reset(seed=args.seed*10000+episode+5000)
                        episode_return = 0.
                        replay = Replay(5000, cfg.n_step, cfg.discount, args.seed+stage)
                    save('latest.pt')
                    if panel:
                        panel.heading.configure(text='VISUAL NETWORK TRAINING - demonstrations + DrQ-v2')
                if panel:
                    time.sleep(max(0., 1./ecfg.policy_hz-(time.monotonic()-tick)))
            save('latest.pt')
        final = {"steps": completed_steps, "updates": agent.updates, "bc_updates": bc_updates,
                 "episodes_completed": episode, "training_successes": successes, "stage": stage,
                 "wall_seconds": time.monotonic()-start, "method": METHOD,
                 "status": "stopped" if panel and panel.stop else "completed", "validations": scores}
        (args.output/'summary.json').write_text(json.dumps(final, indent=2))
        if panel and not panel.stop:
            panel.show_completed(final)
            if args.auto_next:
                panel.heading.configure(text='THIS RUN FINISHED - next run will open after 10 seconds')
                panel.hold_result(10.)
                return
            from .review import run_policy_review
            while not panel.stop:
                panel.pump()
                if panel.replay_requested:
                    panel.output = args.output/'desktop_test_after_training'
                    panel.output.mkdir(exist_ok=True)
                    run_policy_review(panel, env, agent)
                time.sleep(.05)
    except Exception as exc:
        save('interrupted.pt')
        (args.output/'failure.json').write_text(json.dumps(dict(step=completed_steps, reason=repr(exc)), indent=2))
        raise
    finally:
        env.close()
        if panel:
            panel.close()


if __name__ == '__main__':
    main()
