"""Evaluate any saved visual policy on explicit difficulty profiles."""
import argparse
import json
from pathlib import Path

from .agent import DrQV2
from .config import EnvConfig
from .guided import evaluate, stage_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed', type=int, default=60001)
    p.add_argument('--stages', type=int, nargs='*', default=[1, 2])
    p.add_argument('--original', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    agent, metadata = DrQV2.load(args.checkpoint)
    results = dict(checkpoint=str(args.checkpoint), training_metadata=metadata.get('training_metadata'),
                   seed=args.seed, episodes_per_profile=args.episodes, results={})
    profiles = {f'stage_{stage}':stage_config(stage) for stage in args.stages}
    if args.original:
        profiles['original_paper_task'] = EnvConfig()
    for name, config in profiles.items():
        score = evaluate(agent, config, args.episodes, args.seed)
        results['results'][name] = score
        (args.output/'results.json').write_text(json.dumps(results, indent=2))
        print(f'{name}: {score["successes"]}/{score["episodes"]}', flush=True)


if __name__ == '__main__':
    main()
