"""Check response to distinct real images while proprioception stays fixed."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .agent import DrQV2
from .environment import MovingPlatformLanding


def audit(checkpoint):
    agent, _ = DrQV2.load(checkpoint)
    env = MovingPlatformLanding()
    features, actions, images = [], [], []
    try:
        for xyz in ([0, 0, 1.3], [.3, 0, 1.3], [3, 0, 1.3]):
            obs, _ = env.reset(seed=1, options={"initial_xyz": xyz, "platform_speed": 0})
            assert np.allclose(obs["proprio"], np.tile([0, 0, 0, 0, 0, 0, 1], 3))
            with torch.no_grad():
                features.append(agent.encoder(torch.from_numpy(obs["pixels"]).unsqueeze(0)).numpy())
            actions.append(agent.act(obs, evaluation=True))
            images.append(obs["pixels"].astype(float))
    finally:
        env.close()
    return {"checkpoint": str(checkpoint), "real_image_mean_abs_difference": float(np.abs(images[0] - images[1]).mean()),
            "encoder_feature_mean_abs_difference": float(np.abs(features[0] - features[1]).mean()),
            "encoder_feature_abs_mean": float(np.abs(features[0]).mean()),
            "actor_max_action_image_difference": float(np.max(np.abs(np.array(actions) - actions[0]))),
            "actions": [a.tolist() for a in actions],
            "note": "Input-sensitivity diagnostic, not a landing performance measure."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    report = audit(args.checkpoint)
    (args.checkpoint.parent / "image_sensitivity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
