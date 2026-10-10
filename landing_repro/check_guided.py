"""Integration checks for the assisted learner, rewards and truth-free evaluation."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from .agent import DrQV2
from .config import EnvConfig
from .environment import MovingPlatformLanding
from .guided import Demonstrations, clone_step, evaluate, stage_config, teacher_action


def main():
    out = Path('runs/guided_checks')
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    env = MovingPlatformLanding(replace(stage_config(1), lost_view_seconds=.1))
    try:
        env.reset(seed=1, options={'initial_xyz':[4,0,1.3], 'platform_speed':0})
        for _ in range(3):
            _, _, term, trunc, info = env.step(np.zeros(3, np.float32))
        assert term and info['event'] == 'lost_view', info
        results['lost_view_termination'] = 'passed'
        env.cfg = EnvConfig()
        env.reset(seed=1, options={'initial_xyz':[4,0,1.3], 'platform_speed':0})
        for _ in range(3):
            _, _, term, trunc, info = env.step(np.zeros(3, np.float32))
        assert not term and not trunc
        results['original_mode_unchanged'] = 'passed'
        env.cfg = stage_config(1)
        env.reset(seed=1, options={'initial_xyz':[0,0,1.3], 'platform_speed':.2, 'platform_yaw':0})
        for _ in range(300):
            _, reward, term, trunc, info = env.step(teacher_action(env))
            if term or trunc:
                break
        assert info['event'] == 'success' and reward == 10., info
        results['guided_terminal_reward_diagnostic_teacher_only'] = 'passed'
        env.cfg = stage_config(3)
        env.reset(seed=1, options={'initial_xyz':[0,0,4.3], 'platform_speed':0, 'platform_yaw':0})
        assert np.isclose(teacher_action(env)[2]*.5, -.48)
        env.reset(seed=1, options={'initial_xyz':[0,0,1.3], 'platform_speed':0, 'platform_yaw':0})
        assert np.isclose(teacher_action(env)[2]*.5, -.35)
        results['high_stage_descent_schedule'] = 'passed'
    finally:
        env.close()
    agent, metadata = DrQV2.load('runs/guided_pretrain_seed1/latest.pt')
    data = Demonstrations.load('runs/guided_pretrain_seed1/demonstrations.npz')
    expert = data.sample(4)
    batch = {**expert, 'next_pixels':expert['pixels'], 'next_proprio':expert['proprio'],
             'reward':np.full((4,1),.1,np.float32), 'discount':np.full((4,1),.99,np.float32)}
    before = agent.actor.policy[-1].weight.detach().clone()
    metrics = agent.update(batch, 0, expert=expert, bc_weight=5., rl_weight=.02, gradient_clip=5.)
    metrics.update(clone_step(agent, expert))
    assert all(np.isfinite(v) for v in metrics.values())
    assert not torch.equal(before, agent.actor.policy[-1].weight)
    results['real_guided_gradient_updates'] = metrics
    obs = {'pixels':data.pixels[0], 'proprio':data.proprio[0]}
    expected = agent.act(obs, evaluation=True)
    agent.save(out/'model.pt', stage_config(1), 1, {'integration_check':True})
    loaded, meta = DrQV2.load(out/'model.pt')
    assert np.array_equal(expected, loaded.act(obs, evaluation=True))
    assert meta['training_metadata']['integration_check']
    results['checkpoint_reload_exact'] = True
    # If evaluation accidentally executes/queries the teacher this check fails.
    with patch('landing_repro.guided.teacher_action', side_effect=AssertionError('Teacher leaked into evaluation')):
        score = evaluate(loaded, stage_config(0), 1, 62001)
    results['teacher_not_called_in_evaluation'] = True
    observations = [{'pixels':data.pixels[i], 'proprio':data.proprio[0]} for i in np.linspace(0,data.size-1,20,dtype=int)]
    actions = np.stack([loaded.act(o,evaluation=True) for o in observations])
    results['fixed_own_state_image_action_range'] = float(np.ptp(actions,axis=0).max())
    assert results['fixed_own_state_image_action_range'] > .001
    results['status'] = 'all checks passed'
    (out/'checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
