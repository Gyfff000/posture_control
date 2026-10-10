"""Keep completed training results open and inspect the saved model in real physics."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from .agent import DrQV2
from .config import EnvConfig
from .environment import MovingPlatformLanding
from .live_training import LiveTraining


def run_policy_review(panel, env, agent, episodes=3, seed=10001):
    panel.begin_review()
    panel.replay_requested = False
    start = time.monotonic()
    rows = []
    total_steps = successes = 0
    updates_before = agent.updates
    for episode in range(episodes):
        if panel.stop:
            break
        obs, info = env.reset(seed=seed + episode)
        episode_return = 0.
        while not panel.stop:
            panel.wait_if_paused()
            if panel.stop:
                break
            tick = time.monotonic()
            action = agent.act(obs, evaluation=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            episode_return += reward
            panel.update(env, total_steps, episodes * int(env.cfg.episode_seconds * env.cfg.policy_hz),
                         0, agent.updates, episode, successes, 0, episode_return, action, info, {},
                         time.monotonic() - start, force=terminated or truncated)
            if terminated or truncated:
                successes += int(info['success'])
                row = {"episode": episode+1, "seed": seed+episode, "event": info['event'],
                       "success": info['success'], "return": episode_return,
                       "final_xy_error_m": float(np.linalg.norm(info['relative_position'][:2])),
                       "time_s": info['time_s']}
                rows.append(row)
                (panel.output / "test_results.json").write_text(json.dumps({
                    "mode": "new deterministic evaluation of saved model, not recorded training playback",
                    "episodes_completed": len(rows), "successes": successes,
                    "updates_before": updates_before, "updates_after": agent.updates, "episodes": rows}, indent=2))
                panel.episode_result(episode+1, info, episode_return)
                print("Saved model TEST:", json.dumps(row), flush=True)
                panel.hold_result(2.)
                break
            time.sleep(max(0., 1. / env.cfg.policy_hz - (time.monotonic() - tick)))
    assert agent.updates == updates_before, "Review must not update the model"
    panel.review_finished(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    agent, metadata = DrQV2.load(args.run / "latest.pt")
    cfg = EnvConfig(**metadata["env_config"])
    out = args.run / ("desktop_test_" + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True)
    env = MovingPlatformLanding(cfg, gui=True)
    panel = LiveTraining(out)
    panel.show_completed(summary)
    with (args.run / "episodes.csv").open() as f:
        for row in csv.DictReader(f):
            panel.episode_result(int(row['episode']), row, float(row['return']))
    try:
        run_policy_review(panel, env, agent, args.episodes)
        while not panel.stop:
            panel.pump()
            if panel.replay_requested:
                run_policy_review(panel, env, agent, args.episodes)
            time.sleep(.05)
    finally:
        env.close()
        panel.close()


if __name__ == "__main__":
    main()
