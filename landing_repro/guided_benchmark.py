"""Fixed-seed comparisons; evaluation never invokes the training teacher."""
import argparse
import json
from pathlib import Path

import numpy as np

from .agent import DrQV2
from .config import EnvConfig
from .guided import evaluate, stage_config


class ConstantDescent:
    def act(self, observation, evaluation=True):
        return np.array([0., 0., -.7], np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=60001)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    student, metadata = DrQV2.load(a.checkpoint)
    baseline, _ = DrQV2.load(a.baseline)
    profiles = {'near_moving':stage_config(1), 'medium_moving':stage_config(2), 'original_paper_task':EnvConfig()}
    models = {'assisted_visual_network':student, 'old_12000_step_model':baseline, 'constant_descent':ConstantDescent()}
    results = dict(checkpoint=str(a.checkpoint), old_checkpoint=str(a.baseline),
                   training_metadata=metadata.get('training_metadata'), fixed_seeds=[a.seed,a.seed+a.episodes-1],
                   note='New/old models and constant baseline use identical configurations and seeds within each profile. Near/medium profiles are easier than the original paper task.', results={})
    for model_name, model in models.items():
        results['results'][model_name] = {}
        for profile, cfg in profiles.items():
            if model_name == 'constant_descent' and profile == 'original_paper_task':
                continue
            print('TEST', model_name, profile, flush=True)
            score = evaluate(model, cfg, a.episodes, a.seed)
            results['results'][model_name][profile] = score
            (a.output/'results.json').write_text(json.dumps(results, indent=2))
            print(f'{model_name} {profile}: {score["successes"]}/{score["episodes"]}', flush=True)
    print('Benchmark completed', flush=True)


if __name__ == '__main__':
    main()
