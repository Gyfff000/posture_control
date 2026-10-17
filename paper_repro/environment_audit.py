"""CPU physics acceptance checks; injected contact cases are not learned landings."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pybullet as p
from .environment import PaperLanding
from .protocols import ladosz_configs
from .train import write_json
from landing_repro.environment import reward_terms,LIMITS


def pose(env,position,quaternion=(0,0,0,1),velocity=(0,0,0)):
    p.resetBasePositionAndOrientation(env.drone,position,quaternion,physicsClientId=env.client)
    p.resetBaseVelocity(env.drone,velocity,[0,0,0],physicsClientId=env.client)
    env.sim._updateAndStoreKinematicInformation()
    p.performCollisionDetection(physicsClientId=env.client)


def check_contacts(env):
    rows=[]
    for yaw in (0.,np.pi/4):
        env.reset(seed=31,options={'platform_speed':0.,'platform_yaw':yaw})
        for label,local_xy,expected in (
            ('center',(0.,0.),'success'),('edge',(.49,0.),'platform_edge'),
            ('off_platform',(2.,0.),'ground')):
            rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            xy=rotation@local_xy
            pose(env,[*xy,1.])
            lowest=min(p.getAABB(env.drone,i,physicsClientId=env.client)[0][2]
                       for i in range(-1,p.getNumJoints(env.drone,physicsClientId=env.client)))
            surface=0. if label=='off_platform' else env.cfg.platform_top
            pose(env,[*xy,surface+1.-lowest-.001])
            event=env._event()
            row=dict(case=label,yaw_rad=yaw,event=event,expected=expected,
                     platform_contacts=len(p.getContactPoints(env.drone,env.platform_id,physicsClientId=env.client)),
                     ground_contacts=len(p.getContactPoints(env.drone,env.sim.PLANE_ID,physicsClientId=env.client)))
            rows.append(row)
            assert event==expected,row
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/paper_audit_20261017/environment_acceptance.json'))
    args=parser.parse_args()
    cfg,_=ladosz_configs()
    env=PaperLanding(cfg);checks={}
    try:
        obs,_=env.reset(seed=41,options={'initial_xyz':[0,0,3.],'platform_speed':.6,'platform_yaw':.7})
        np.testing.assert_array_equal(obs['pixels'][0],obs['pixels'][2])
        np.testing.assert_array_equal(obs['proprio'][:7],obs['proprio'][-7:])
        # Known world velocity and quaternion must preserve axes and order.
        quat=p.getQuaternionFromEuler([.1,-.2,.3]);velocity=np.array([.3,-.6,.1])
        pose(env,[0,0,3.],quat,velocity);env._observe()
        expected=(np.r_[velocity/LIMITS,quat]+1.)*.5
        np.testing.assert_allclose(env._observation()['proprio'][-7:],expected,atol=1e-6)
        checks['world_velocity_xyzw_encoding']=True

        # Moving target may alter pixels, never any of the 21 own-state entries.
        for _ in range(3):env._observe()
        own=env._observation()['proprio'].copy();pixels=env._observation()['pixels'].copy()
        target=env.platform_position
        p.resetBasePositionAndOrientation(env.platform_id,target+[3,0,0],
            p.getBasePositionAndOrientation(env.platform_id,physicsClientId=env.client)[1],physicsClientId=env.client)
        for _ in range(3):env._observe()
        np.testing.assert_array_equal(own,env._observation()['proprio'])
        assert not np.array_equal(pixels,env._observation()['pixels'])
        checks['target_truth_excluded_from_entire_own_history']=True
        checks['camera_responds_to_target_motion']=True
        obs,_=env.reset(seed=42,options={'initial_xyz':[0,0,3.],'platform_speed':0.})
        for image in obs['pixels']:np.testing.assert_array_equal(image,obs['pixels'][0])
        for own_frame in obs['proprio'].reshape(3,7):np.testing.assert_array_equal(own_frame,obs['proprio'][:7])
        checks['episode_reset_clears_histories']=True
        before=obs['pixels'][-1].copy()
        pose(env,[0,0,3.],p.getQuaternionFromEuler([.4,0,0]));env._observe()
        assert not np.array_equal(before,env._observation()['pixels'][-1])
        checks['body_fixed_camera_responds_to_attitude']=True

        env.reset(seed=43,options={'initial_xyz':[0,0,10.],'platform_speed':.6,'platform_yaw':.7})
        origin=env.platform_position.copy();velocity=env.platform_velocity.copy()
        for k in range(30):
            _,_,terminated,truncated,info=env.step([0,0,0])
            assert not terminated and not truncated
            np.testing.assert_allclose(env.platform_position,origin+velocity*env.elapsed,atol=1e-6)
            assert abs(env.elapsed-(k+1)/30.)<1e-8
        checks['constant_motion_world_axes_30_steps']=True

        contacts=check_contacts(env)
    finally:env.close()
    # A shortened physical episode tests timeout vs collision and reset enforcement.
    short=PaperLanding(replace(cfg,episode_seconds=.1))
    try:
        short.reset(seed=44,options={'initial_xyz':[0,0,10.],'platform_speed':0.})
        for _ in range(3):_,_,term,trunc,info=short.step([0,0,0])
        assert not term and trunc and info['event']=='timeout'
        try:short.step([0,0,0])
        except RuntimeError:checks['timeout_and_reset_enforced']=True
        else:raise AssertionError('Step accepted after timeout')
    finally:short.close()

    rewards=[]
    for distance in (0.,5.,10.,10.001):
        terms=reward_terms(distance,-.5,-.5,True,'running',cfg)
        rewards.append(dict(distance_m=distance,**terms))
    assert abs(rewards[0]['r_xy']-1.)<1e-9
    assert rewards[2]['r_xy']==0. and rewards[3]['r_xy']==0.
    assert rewards[0]['reward']>rewards[1]['reward']>rewards[2]['reward']
    assert reward_terms(0,-.1,-.1,False,'success',cfg)['reward']==-.01
    assert reward_terms(0,-.1,-.1,True,'success',cfg)['r_landing']==120.
    checks['reward_boundary_and_visibility_semantics']=True
    result=dict(status='passed',checks=checks,injected_contact_cases=contacts,reward_boundaries=rewards,
        interpretation='normalized horizontal reward + actual vertical speed guard; not author-verified',
        limitation='Injected contacts validate event logic, not a trained policy or paper success rate.')
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,result)
    print(json.dumps(result))
if __name__=='__main__':main()
