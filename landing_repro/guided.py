"""Demonstration assisted visual training, explicitly distinct from pure DrQ-v2.

Simulator truth labels actions during training ONLY. Evaluation uses DrQV2.act
with pixels + own state and does not call the teacher.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .agent import DrQV2, Replay
from .config import AgentConfig, EnvConfig
from .environment import MovingPlatformLanding, LIMITS


METHOD = "DrQ-v2 + privileged demonstration pretraining and online imitation; modified reward/curriculum"


def stage_config(stage):
    specs = [
        dict(initial_height=1.0, offset_xy=.45, offset_z=.15, platform_speed_max=0.),
        dict(initial_height=1.5, offset_xy=.65, offset_z=.3, platform_speed_max=.3),
        dict(initial_height=2.5, offset_xy=1., offset_z=.5, platform_speed_max=.5),
        dict(initial_height=5., offset_xy=2., offset_z=1., platform_speed_max=.7),
        dict(initial_height=10., offset_xy=3., offset_z=3., platform_speed_max=1.)]
    return EnvConfig(**specs[stage], platform_yaw_degrees=180., episode_seconds=40.,
                     reward_mode="guided_progress", escape_distance=12., lost_view_seconds=1.,
                     teacher_descent_fast=.48 if stage >= 3 else .35)


def teacher_action(env):
    error = env.platform_position[:2] - env.state[:2]
    xy = env.platform_velocity[:2] + 2.0 * error
    # Smooth descent target avoids a discontinuous label near the alignment threshold.
    distance = np.linalg.norm(error)
    height = env.state[2] - env.cfg.platform_top
    descent = env.cfg.teacher_descent_fast if height > 1.5 else .35
    vz = -descent * np.clip((.38-distance)/.22, 0., 1.)
    return np.clip(np.r_[xy, vz] / LIMITS, -1, 1).astype(np.float32)


class Demonstrations:
    def __init__(self, capacity=12000, seed=1):
        self.capacity, self.size, self.index = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.pixels = np.empty((capacity, 3, 84, 84), np.uint8)
        self.proprio = np.empty((capacity, 21), np.float32)
        self.actions = np.empty((capacity, 3), np.float32)

    def add(self, obs, action):
        i = self.index
        self.pixels[i], self.proprio[i], self.actions[i] = obs['pixels'], obs['proprio'], action
        self.index = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, count):
        i = self.rng.integers(self.size, size=count)
        return {"pixels": self.pixels[i], "proprio": self.proprio[i], "action": self.actions[i]}

    def save(self, path):
        np.savez_compressed(path, pixels=self.pixels[:self.size], proprio=self.proprio[:self.size],
                            action=self.actions[:self.size])

    @classmethod
    def load(cls, path, seed=1):
        data = np.load(path)
        obj = cls(max(12000, len(data['action'])), seed)
        obj.size = len(data['action'])
        obj.pixels[:obj.size], obj.proprio[:obj.size], obj.actions[:obj.size] = data['pixels'], data['proprio'], data['action']
        obj.index = obj.size % obj.capacity
        return obj


def clone_step(agent, batch):
    features = agent.encoder(torch.as_tensor(batch['pixels']))
    features = torch.cat([features, torch.as_tensor(batch['proprio'])], dim=-1)
    predicted = agent.actor(features)
    target = torch.as_tensor(batch['action'])
    loss = ((predicted-target).square() * torch.tensor([3., 3., 1.])).mean()
    for i in (0, 1):
        agent.optimizers[i].zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.encoder.parameters(), 5.)
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 5.)
    for i in (0, 1):
        agent.optimizers[i].step()
    return {"bc_loss": float(loss.detach())}


def collect_demonstrations(output, episodes=40, seed=1, max_stage=2):
    data = Demonstrations(seed=seed)
    rows = []
    env = MovingPlatformLanding(stage_config(0))
    try:
        for episode in range(episodes):
            stage = episode % (max_stage+1)
            env.cfg = stage_config(stage)
            obs, _ = env.reset(seed=seed*1000+episode)
            steps = 0
            while True:
                action = teacher_action(env)
                data.add(obs, action)
                obs, reward, term, trunc, info = env.step(action)
                steps += 1
                if term or trunc:
                    break
            rows.append(dict(episode=episode+1, stage=stage, event=info['event'], steps=steps))
            print(f"Teacher DATA ONLY episode={episode+1}/{episodes} stage={stage} event={info['event']} samples={data.size}", flush=True)
    finally:
        env.close()
    data.save(output / 'demonstrations.npz')
    (output / 'teacher_collection.json').write_text(json.dumps(rows, indent=2))
    return data


def evaluate(agent, config, episodes=10, seed=20001, progress=None, observer=None):
    # This function intentionally has no reference to teacher_action.
    env = MovingPlatformLanding(config)
    rows = []
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed+episode)
            total = 0.
            step = 0
            while True:
                action = agent.act(obs, evaluation=True)
                obs, reward, term, trunc, info = env.step(action)
                total += reward
                step += 1
                if observer:
                    observer(env, episode+1, episodes, info, action, total, term or trunc)
                if progress and (step % 30 == 0 or term or trunc):
                    progress()
                if term or trunc:
                    break
            rows.append(dict(seed=seed+episode, success=info['success'], event=info['event'],
                             xy_error=float(np.linalg.norm(info['relative_position'][:2])), return_=total,
                             platform_speed_m_s=float(np.linalg.norm(env.platform_velocity))))
    finally:
        env.close()
    return dict(episodes=episodes, successes=sum(r['success'] for r in rows),
                success_rate=sum(r['success'] for r in rows)/episodes, config=config.to_dict(), rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--demo-episodes', type=int, default=40)
    parser.add_argument('--bc-updates', type=int, default=2000)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--max-stage', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = AgentConfig(seed=args.seed, learning_rate=.0001, batch_size=32, hidden_dim=256,
                      noise_start=.12, noise_end=.03, noise_decay_steps=40000, tau=.01)
    (args.output/'config.json').write_text(json.dumps(dict(method=METHOD, agent=cfg.to_dict(),
        note="Teacher uses simulator truth for training labels; evaluation policy receives pixels/own state only"), indent=2))
    data = Demonstrations.load(args.data, args.seed) if args.data else collect_demonstrations(args.output, args.demo_episodes, args.seed, args.max_stage)
    agent = DrQV2(cfg)
    start = time.monotonic()
    history = []
    with (args.output/'bc_metrics.jsonl').open('w') as log:
        for update in range(1, args.bc_updates+1):
            metrics = clone_step(agent, data.sample(cfg.batch_size))
            log.write(json.dumps(dict(update=update, **metrics))+'\n')
            if update % 100 == 0:
                log.flush()
                print(f"VISUAL NETWORK BC update={update}/{args.bc_updates} loss={metrics['bc_loss']:.5f} elapsed={time.monotonic()-start:.1f}s", flush=True)
            if update % args.eval_every == 0 or update == args.bc_updates:
                scores = {str(stage): evaluate(agent, stage_config(stage), 5) for stage in range(args.max_stage+1)}
                history.append(dict(bc_updates=update, scores=scores))
                (args.output/'evaluations.json').write_text(json.dumps(history, indent=2))
                agent.save(args.output/'latest.pt', stage_config(1), 0,
                    dict(method=METHOD, phase="demonstration pretraining", bc_updates=update, seed=args.seed))
                print("HELD-OUT NETWORK ONLY:", {k:f"{v['successes']}/{v['episodes']}" for k,v in scores.items()}, flush=True)
    print('Pretraining completed', flush=True)


if __name__ == '__main__':
    main()
