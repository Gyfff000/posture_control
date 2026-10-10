"""Checks for action signs, contact events, camera geometry and bootstrap."""
import json
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

from .agent import Replay
from .config import EnvConfig
from .environment import MovingPlatformLanding, reward_terms, LIMITS


def oracle_action(env):
    """Truth-based controller ONLY for validating the environment, never DrQ data."""
    error = env.platform_position[:2] - env.state[:2]
    xy = env.platform_velocity[:2] + 2.5 * error
    vz = -.45 if np.linalg.norm(error) < .2 else 0.
    return np.clip(np.r_[xy, vz] / LIMITS, -1, 1).astype(np.float32)


def main():
    out = Path("runs/validation")
    out.mkdir(parents=True, exist_ok=True)
    cfg = EnvConfig()
    r0 = reward_terms(0, -.5, -.5, True, "running", cfg)
    rb = reward_terms(10, 0, 0, True, "running", cfg)
    assert np.isclose(r0["r_xy"], 1) and np.isclose(r0["r_z"], 1)
    assert np.isclose(rb["r_xy"], 0) and np.isclose(rb["r_z"], 0)
    assert reward_terms(0, 0, .2, True, "running", cfg)["r_z"] == -.01
    assert reward_terms(0, -8, -.5, True, "running", cfg)["r_z"] == -.01
    assert reward_terms(0, -.5, -.5, False, "running", cfg)["reward"] == -.01
    dummy = {"pixels": np.zeros((3, 84, 84), np.uint8), "proprio": np.zeros(21, np.float32)}
    for terminal, expected in [(False, .25), (True, 0.)]:
        replay = Replay(4, n_step=3, discount=.5)
        replay.add(dummy, np.zeros(3), 1, dummy, False, False)
        replay.add(dummy, np.zeros(3), 2, dummy, terminal, not terminal)
        assert replay.size == 2 and np.isclose(replay.data["reward"][0, 0], 2)
        assert np.isclose(replay.data["discount"][0, 0], expected)
    env = MovingPlatformLanding()
    report = {"reward_boundaries": "passed", "terminal_timeout_bootstrap": "passed"}
    try:
        for label, xyz in [("high", [0, 0, 10.3]), ("near", [0, 0, 1.3]), ("out_of_view", [5, 0, 1.3])]:
            obs, info = env.reset(seed=1, options={"initial_xyz": xyz, "platform_speed": 0})
            assert env.observation_space.contains(obs)
            assert np.array_equal(obs["pixels"][0], obs["pixels"][2])
            Image.fromarray(env.rgb).resize((420, 420), Image.Resampling.NEAREST).save(out / f"camera_{label}.png")
            report[label + "_visible"] = info["visible"]
        assert report["near_visible"] and not report["out_of_view_visible"]
        env.reset(seed=1, options={"initial_xyz": [0, 0, 2], "platform_speed": .4})
        origin = env.platform_position.copy()
        for _ in range(60):
            env.step(np.array([1., 0., 0.], np.float32))
        expected = origin + env.platform_velocity * env.elapsed
        motion_error = float(np.linalg.norm(env.platform_position - expected))
        assert motion_error < 1e-6, motion_error
        assert env.state[10] > 1.0 and abs(env.state[12]) < .15, env.state[10:13]
        report.update(platform_motion_error_m=motion_error, positive_x_velocity_m_s=float(env.state[10]),
                      measured_platform_velocity=p.getBaseVelocity(env.platform_id, physicsClientId=env.client)[0])
        env.reset(seed=2, options={"initial_xyz": [0, 0, 1.3], "platform_speed": 0})
        initial_view = np.array(env.camera_view)
        pos = env.state[:3]
        p.resetBasePositionAndOrientation(env.drone, pos, p.getQuaternionFromEuler([0, .4, 0]), physicsClientId=env.client)
        env.sim._updateAndStoreKinematicInformation()
        env._observe()
        assert not np.allclose(env.camera_view, initial_view)
        report["body_fixed_camera"] = "passed"
        env.reset(seed=3, options={"initial_xyz": [0, 0, 1.3], "platform_speed": .4})
        for _ in range(300):
            _, _, term, trunc, info = env.step(oracle_action(env))
            if term or trunc:
                break
        assert info["event"] == "success", info
        assert info["reward"] == 120., info
        report["moving_platform_contact"] = info
        Image.fromarray(env.rgb).resize((420, 420), Image.Resampling.NEAREST).save(out / "camera_contact.png")
        env.reset(seed=4, options={"initial_xyz": [2, 0, .5], "platform_speed": 0})
        for _ in range(150):
            _, _, term, trunc, info = env.step(np.array([0, 0, -.8], np.float32))
            if term or trunc:
                break
        assert info["event"] == "ground", info
        report["ground_collision"] = info["event"]
        env.reset(seed=5, options={"initial_xyz": [.49, 0, .6], "platform_speed": 0})
        for _ in range(150):
            _, _, term, trunc, info = env.step(np.array([0, 0, -.8], np.float32))
            if term or trunc:
                break
        assert info["event"] == "platform_edge", info
        report["edge_rejected"] = info["event"]
    finally:
        env.close()
    env = MovingPlatformLanding(EnvConfig(episode_seconds=.1))
    try:
        env.reset(seed=6)
        for _ in range(3):
            _, _, term, trunc, info = env.step(np.zeros(3))
        assert trunc and not term and info["event"] == "timeout"
        report["time_limit"] = "passed"
    finally:
        env.close()
    report["status"] = "all checks passed"
    (out / "checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
