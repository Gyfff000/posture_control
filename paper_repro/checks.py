"""Meaningful physics, policy/replay equivalence and information-boundary checks."""
from pathlib import Path
from dataclasses import replace
import json,tempfile
import numpy as np
import pybullet as p
import torch
from landing_repro.agent import Replay,Encoder
from landing_repro.environment import reward_terms
from .agent import PaperAgent,CheckpointEncoder
from .environment import PaperLanding
from .protocols import ladosz_configs,PERCHRL_PAPER,LADOSZ_PAPER
from .replay import DiskReplay
from .perchrl_components import PerchPolicy,reliability,active_reward,sample_initialization

def main():
    out=Path('results/paper_audit_20261017');out.mkdir(exist_ok=True,parents=True)
    cfg,acfg=ladosz_configs();env=PaperLanding(cfg);rows=[]
    with tempfile.TemporaryDirectory() as temp:
        regular=Replay(32,3,.99,17);disk=DiskReplay(Path(temp)/'replay',32,3,.99,17)
        try:
            obs,_=env.reset(seed=22)
            assert obs['pixels'].shape==(3,84,84) and obs['proprio'].shape==(21,)
            assert np.min(obs['proprio'])>=0 and np.max(obs['proprio'])<=1
            assert not hasattr(env,'_center_tracker') and not cfg.prediction_guidance
            for seed in range(10):
                obs,_=env.reset(seed=seed)
                h=env.state[2]-cfg.platform_top
                assert 7.<=h<=13.
                assert np.max(np.abs(env.state[:2]))<=3.
                assert np.linalg.norm(env.platform_velocity)<=1.00001
                rows.append(dict(seed=seed,height_m=h,platform_velocity=env.platform_velocity.tolist()))
            own=env._observation()['proprio'].copy();position,quat=p.getBasePositionAndOrientation(env.platform_id,physicsClientId=env.client)
            p.resetBasePositionAndOrientation(env.platform_id,np.asarray(position)+[10,10,0],quat,physicsClientId=env.client)
            env._observe();np.testing.assert_array_equal(own[-7:],env._observation()['proprio'][-7:])
            obs,_=env.reset(seed=51)
            for k in range(36):
                action=np.array([.05,-.05,-.5],np.float32)
                nxt,r,term,trunc,info=env.step(action)
                assert np.isfinite(r) and abs(env.elapsed-(k+1)/30.)<1e-7
                np.testing.assert_allclose(info['applied_velocity'],action*[1.5,1.5,.5],atol=1e-6)
                regular.add(obs,action,r,nxt,term,trunc);disk.add(obs,action,r,nxt,term,trunc);obs=nxt
            for key in regular.data:np.testing.assert_array_equal(regular.data[key],disk.data[key])
            expected=regular.sample(8);actual=disk.sample(8)
            for key in expected:np.testing.assert_array_equal(expected[key],actual[key])
            disk.flush();restored=DiskReplay(Path(temp)/'replay',32,3,.99,17,resume=True)
            expected=disk.sample(8);actual=restored.sample(8)
            for key in expected:np.testing.assert_array_equal(expected[key],actual[key])
            model=PaperAgent(acfg);before=model.act(obs,evaluation=True)
            metrics=model.update(restored.sample(16),10000)
            assert np.isfinite(list(metrics.values())).all() and metrics['imitation_loss']==0 and metrics['smooth_loss']==0
            model.save(Path(temp)/'model.pt',cfg,36,dict(method='check'))
            loaded,_=PaperAgent.load(Path(temp)/'model.pt')
            np.testing.assert_allclose(model.act(obs,evaluation=True),loaded.act(obs,evaluation=True),atol=1e-7)
            assert not np.allclose(before,model.act(obs,evaluation=True))
            invisible=reward_terms(0.,-.1,-.1,False,'success',cfg)
            assert invisible['reward']==-.01
            literal=reward_terms(0.,-.1,-.1,True,'running',replace(cfg,horizontal_reward='literal'))
            assert literal['r_xy']>1e10  # Audit the paper ambiguity, never silently relabel it.
            literal_v=reward_terms(0.,-8.,-.2,True,'running',replace(cfg,guard_actual_vertical_velocity=False))
            guarded_v=reward_terms(0.,-8.,-.2,True,'running',cfg)
            assert literal_v['reward']>1e20 and guarded_v['reward']<1.
            assert cfg.guard_actual_vertical_velocity
        finally:env.close()
    # Activation checkpointing changes memory scheduling, not the 1024-sample BN batch.
    torch.manual_seed(7);standard=Encoder();chunked=CheckpointEncoder();chunked.load_state_dict(standard.state_dict())
    inputs=torch.randint(0,256,(64,3,84,84),dtype=torch.uint8)
    y1=standard(inputs);y2=chunked(inputs);torch.testing.assert_close(y1,y2,atol=1e-5,rtol=1e-4)
    y1.square().mean().backward();y2.square().mean().backward()
    for p1,p2 in zip(standard.parameters(),chunked.parameters()):torch.testing.assert_close(p1.grad,p2.grad,atol=1e-5,rtol=1e-3)
    torch.manual_seed(1);policy=PerchPolicy(history_length=30)
    history=torch.randn(4,30,10);own=torch.randn(4,12);rho=torch.ones(4,1);previous=torch.zeros(4,4);future=torch.randn(4,15,10)
    action=policy.action(history,own,rho,previous)
    assert action.shape==(4,4) and policy.value(history,own,rho,previous,future).shape==(4,1)
    with torch.no_grad():
        a=policy.action(history,own,rho,previous);policy.value(history,own,rho,previous,future*100)
        torch.testing.assert_close(a,policy.action(history,own,rho,previous))
    np.testing.assert_allclose(reliability(np.array([0.,1.,2.]),1.),np.exp(-np.array([0.,1.,2.])))
    assert active_reward([0,0,0],[0,0,0],0.,[0,0,1],[0,0,1],1,1,1,1)==0.
    samples=sample_initialization(np.random.default_rng(42),1000)
    for key,(lo,hi) in PERCHRL_PAPER['initialization'].items():assert np.all((samples[key]>=lo)&(samples[key]<=hi))
    result=dict(status='passed',ladosz_resets=rows,physics_steps=36,disk_replay_equivalence=True,
        disk_replay_resume=True,encoder_checkpoint_gradient_equivalence=True,policy_optimizer_update=True,checkpoint_reload=True,
        no_assisted_control=True,no_platform_truth_in_proprio=True,
        perchrl_components={'initialization_samples':1000,'actor_shape':[4,4],
            'future_inputs_critic_only':True,'history_length_30':'implementation choice, not verified from paper'},
        numerical_reproduction_complete=False)
    (out/'component_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    (out/'ladosz2024_published.json').write_text(json.dumps(LADOSZ_PAPER,indent=2))
    (out/'perchrl2026_published.json').write_text(json.dumps(PERCHRL_PAPER,indent=2))
if __name__=='__main__':main()
