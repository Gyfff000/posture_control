import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .agent import DrQV2
from .checks import oracle_action
from .config import EnvConfig
from .environment import MovingPlatformLanding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--controller", choices=["drq", "oracle", "zero"], default="drq")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    args = parser.parse_args()
    if args.controller == "drq" and not args.checkpoint:
        parser.error("DrQ evaluation requires --checkpoint")
    cfg = EnvConfig()
    agent = None
    if args.checkpoint:
        agent, metadata = DrQV2.load(args.checkpoint)
        cfg = EnvConfig(**metadata["env_config"])
    env = MovingPlatformLanding(cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode)
            total, invisible, steps = 0., 0, 0
            while True:
                action = agent.act(obs, evaluation=True) if args.controller == "drq" else (
                    oracle_action(env) if args.controller == "oracle" else np.zeros(3, np.float32))
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                invisible += not info["visible"]
                if terminated or truncated:
                    break
            row = {"episode": episode + 1, "seed": args.seed + episode, "controller": args.controller,
                   "event": info["event"], "success": info["success"], "return": total, "steps": steps,
                   "time_s": info["time_s"], "final_xy_error_m": float(np.linalg.norm(info["relative_position"][:2])),
                   "final_height_above_pad_m": info["relative_position"][2],
                   "final_relative_speed_m_s": float(np.linalg.norm(info["relative_velocity"])),
                   "invisible_fraction": invisible / steps, "platform_speed_m_s": float(np.linalg.norm(env.platform_velocity)),
                   "final_rpy": info["rpy"]}
            rows.append(row)
            print(f"episode={episode+1} controller={args.controller} event={info['event']} "
                  f"xy_error={row['final_xy_error_m']:.3f} height={row['final_height_above_pad_m']:.3f}", flush=True)
    finally:
        env.close()
    with (args.output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"controller": args.controller, "checkpoint": str(args.checkpoint), "episodes": len(rows),
               "successes": sum(r["success"] for r in rows),
               "success_rate": sum(r["success"] for r in rows) / len(rows),
               "events": {event: sum(r["event"] == event for r in rows) for event in {r["event"] for r in rows}},
               "mean_final_xy_error_m": float(np.mean([r["final_xy_error_m"] for r in rows])),
               "note": "Small independent-seed evaluation; not the paper's 500-episode, five-seed comparison.",
               "environment": cfg.to_dict()}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
