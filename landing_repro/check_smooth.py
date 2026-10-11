"""Integration checks for transfer, command memory/jerk bounds and evaluation."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .agent import DrQV2
from .environment import MovingPlatformLanding
from .guided import stage_config
from .smooth import evaluate_flights, smooth_config, transfer


def main():
    out=Path('runs/smooth_checks'); out.mkdir(exist_ok=True)
    source=Path('runs/guided_long_20261010_231249/seed_11/best_stage2.pt')
    old,_=DrQV2.load(source)
    agent,_=transfer(source)
    env=MovingPlatformLanding(smooth_config(1))
    results={}
    try:
        obs,_=env.reset(seed=1,options={'initial_xyz':[0,0,2.3],'platform_speed':0})
        assert obs['proprio'].shape==(30,)
        assert np.all(obs['proprio'][21:]==0)
        a=old.act(dict(pixels=obs['pixels'],proprio=obs['proprio'][:21]),evaluation=True)
        assert np.allclose(a,agent.act(obs,evaluation=True),atol=1e-6)
        results['transfer_preserves_initial_actions']=True
        previous_acc=env.command_acceleration.copy()
        for index in range(60):
            action=np.array([.8 if index<30 else -.8,0.,-.2],np.float32)
            obs,reward,term,trunc,info=env.step(action)
            assert np.allclose(obs['proprio'][21:24],action)
            # Two control ticks per policy tick: bounded total acceleration change.
            assert np.all(np.abs(env.command_acceleration-previous_acc)<=np.array([8.,8.,4.])/30+1e-6)
            assert np.all(np.abs(env.command_acceleration)<=np.array([2.,2.,.8])+1e-6)
            assert info['smooth_penalty']>=0 and np.isfinite(reward)
            previous_acc=env.command_acceleration.copy()
            if term or trunc:
                break
        results['own_command_memory_and_command_jerk_bounds']=True
        env.reset(seed=2)
        assert np.all(env.previous_command==0) and np.all(env.command_acceleration==0)
        results['episode_memory_reset']=True
        expert=dict(pixels=np.stack([obs['pixels']]*2),proprio=np.stack([obs['proprio']]*2),action=np.zeros((2,3),np.float32))
        batch={**expert,'next_pixels':expert['pixels'],'next_proprio':expert['proprio'],
               'reward':np.zeros((2,1),np.float32),'discount':np.full((2,1),.99,np.float32)}
        before=agent.actor.policy[-1].weight.detach().clone()
        metrics=agent.update(batch,0,expert,bc_weight=3.,rl_weight=.02,gradient_clip=5.,smooth_weight=.15)
        assert all(np.isfinite(v) for v in metrics.values()) and metrics['smooth_loss']>=0
        assert not torch.equal(before,agent.actor.policy[-1].weight)
        results['finite_real_smooth_gradient_update']=True
        agent.save(out/'model.pt',smooth_config(1),1,{'test':True})
        loaded,_=DrQV2.load(out/'model.pt')
        assert np.array_equal(agent.act(obs,evaluation=True),loaded.act(obs,evaluation=True))
        results['smooth_checkpoint_reload_exact']=True
        config=smooth_config(1); config.episode_seconds=.2
        with patch('landing_repro.smooth.smooth_teacher',side_effect=AssertionError('teacher used in test')):
            result=evaluate_flights(loaded,config,out/'short_test',1,65001)
        assert result['episodes']==1
        results['no_teacher_no_learning_in_evaluation']=True
        baseline=MovingPlatformLanding(stage_config(1))
        try:
            raw,_=baseline.reset(seed=1)
            assert raw['proprio'].shape==(21,)
            assert baseline.cfg.jerk_limit_xy==0 and baseline.cfg.action_smooth_weight==0
        finally:
            baseline.close()
        results['original_configuration_compatible']=True
    finally:
        env.close()
    (out/'checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
