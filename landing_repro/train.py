"""CPU short training. Use separate evaluation seeds; no success guarantee."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .agent import DrQV2, Replay
from .config import AgentConfig, EnvConfig
from .environment import MovingPlatformLanding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/cpu_short"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--lr", type=float, default=.0001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = AgentConfig(seed=args.seed, batch_size=args.batch_size, update_every=args.update_every,
                      warmup_steps=args.warmup, learning_rate=args.lr)
    ecfg = EnvConfig()
    (args.output / "config.json").write_text(json.dumps({"agent": cfg.to_dict(), "environment": ecfg.to_dict(),
        "purpose": "CPU short training; not paper-scale convergence or published success rate"}, indent=2))
    rng = np.random.default_rng(args.seed)
    env, agent = MovingPlatformLanding(ecfg), DrQV2(cfg)
    replay = Replay(cfg.replay_capacity, cfg.n_step, cfg.discount, args.seed)
    writer = SummaryWriter(str(args.output / "tensorboard"))
    obs, _ = env.reset(seed=args.seed)
    start = time.monotonic()
    episode, episode_return, episode_length, successes = 0, 0., 0, 0
    initial_parameter = agent.actor.policy[-1].weight.detach().clone()
    metrics = {}
    try:
        with (args.output / "metrics.jsonl").open("w") as mf, (args.output / "episodes.csv").open("w", newline="") as ef:
            ew = csv.writer(ef)
            ew.writerow(["episode", "steps", "return", "event", "time_s", "contact_xy_error_m", "relative_speed_m_s"])
            for step in range(args.steps):
                action = rng.uniform(-1, 1, 3).astype(np.float32) if step < cfg.warmup_steps else agent.act(obs, step)
                next_obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(reward):
                    raise FloatingPointError("Non-finite reward")
                replay.add(obs, action, reward, next_obs, terminated, truncated)
                episode_return += reward
                episode_length += 1
                obs = next_obs
                if step >= cfg.warmup_steps and replay.size >= cfg.batch_size and step % cfg.update_every == 0:
                    metrics = agent.update(replay.sample(cfg.batch_size), step)
                    mf.write(json.dumps({"step": step + 1, **metrics}) + "\n")
                    for key, value in metrics.items():
                        writer.add_scalar("train/" + key, value, step + 1)
                if terminated or truncated:
                    episode += 1
                    successes += int(info["success"])
                    ew.writerow([episode, episode_length, episode_return, info["event"], info["time_s"],
                                 np.linalg.norm(info["relative_position"][:2]), np.linalg.norm(info["relative_velocity"])])
                    ef.flush()
                    writer.add_scalar("episode/return", episode_return, step + 1)
                    print(f"episode={episode} event={info['event']} return={episode_return:.2f}", flush=True)
                    obs, _ = env.reset(seed=args.seed + episode)
                    episode_return, episode_length = 0., 0
                if (step + 1) % 200 == 0:
                    mf.flush()
                    print(f"step={step+1}/{args.steps} updates={agent.updates} elapsed={time.monotonic()-start:.1f}s "
                          f"loss={metrics.get('critic_loss', 0):.4f}", flush=True)
                if (step + 1) % 1000 == 0:
                    agent.save(args.output / "latest.pt", ecfg, step + 1)
            agent.save(args.output / "latest.pt", ecfg, args.steps)
            before = agent.act(obs, evaluation=True)
            loaded, _ = DrQV2.load(args.output / "latest.pt")
            after = loaded.act(obs, evaluation=True)
            reload_error = float(np.max(np.abs(before - after)))
            assert reload_error < 1e-7
            change = float((agent.actor.policy[-1].weight.detach() - initial_parameter).abs().max())
            assert agent.updates > 0 and change > 0
            summary = {"steps": args.steps, "updates": agent.updates, "episodes_completed": episode,
                       "training_successes": successes, "wall_seconds": time.monotonic() - start,
                       "parameter_max_change": change, "checkpoint_action_max_error": reload_error,
                       "final_metrics": metrics, "status": "CPU short training completed, convergence not established"}
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        (args.output / "failure.json").write_text(json.dumps({"status": "failed", "step": step + 1,
                                                             "reason": repr(exc)}, indent=2))
        raise
    finally:
        env.close()
        writer.close()


if __name__ == "__main__":
    main()
