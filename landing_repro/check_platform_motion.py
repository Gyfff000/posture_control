"""Check geometric/dynamic bounds, seeded reproducibility and actual Bullet motion."""
import json
from pathlib import Path
import numpy as np
import pybullet as p

from .config import EnvConfig
from .environment import MovingPlatformLanding
from .motion_curriculum import motion_config,promotion_ready
from .platform_motion import PlatformMotion,spline
from .smooth import smooth_config


def main():
    rows=[]
    for phase in (1,2,3):
        cfg=motion_config(smooth_config,phase,True,True,True)
        for seed in range(8):
            speed=.5*(cfg.platform_speed_min+cfg.platform_speed_max)
            motion=PlatformMotion(cfg,np.random.default_rng(seed),[0,0,.26],speed,.7)
            v=motion.velocities; speeds=np.linalg.norm(v[:,:2],axis=1)
            acceleration=np.linalg.norm(np.diff(v,axis=0)/motion.dt,axis=1)
            yaw_rate=np.abs(np.diff(motion.headings)/motion.dt)
            assert speeds.min()>=cfg.platform_speed_min-1e-9 and speeds.max()<=cfg.platform_speed_max+1e-9
            assert acceleration.max()<=cfg.platform_accel_max*1.01
            assert yaw_rate.max()<=np.deg2rad(cfg.platform_turn_rate_degrees)*1.01
            assert np.ptp(speeds)>.005
            if phase>=2:
                assert np.ptp(motion.headings)>.05
            repeat=PlatformMotion(cfg,np.random.default_rng(seed),[0,0,.26],speed,.7)
            assert np.array_equal(motion.positions,repeat.positions)
            assert np.array_equal(motion.velocities,repeat.velocities)
            rows.append(dict(phase=phase,seed=seed,min_speed=float(speeds.min()),max_speed=float(speeds.max()),
                max_acceleration=float(acceleration.max()),max_turn_rate_deg_s=float(np.rad2deg(yaw_rate.max())),
                max_curvature=motion.max_curvature))
    controls=np.random.default_rng(12).normal(size=(12,2))
    values=spline(controls,np.array([0.,12.]))
    assert all(np.allclose(x[0],x[1]) for x in values)  # position, tangent and curvature inputs at seam
    old=EnvConfig()
    linear=PlatformMotion(old,np.random.default_rng(0),[0,0,.26],.2,.7)
    assert np.allclose(linear.sample(7.)[0],[.2*np.cos(.7)*7,.2*np.sin(.7)*7,.26])
    good=dict(episodes=20,successes=17,soft_landings=14,motion_phase=1)
    assert not promotion_ready([good]*2,1,20)
    assert promotion_ready([good]*3,1,20)
    assert not promotion_ready([good]*3,2,20)
    assert not promotion_ready([good,good,dict(good,episodes=19)],1,20)
    cfg=motion_config(smooth_config,2,True,True,True)
    env=MovingPlatformLanding(cfg)
    try:
        obs,_=env.reset(seed=42,options=dict(initial_xyz=[0,0,3.],platform_speed=.25,platform_yaw=.7))
        initial_velocity=env.platform_velocity.copy()
        for _ in range(90):
            obs,_,term,trunc,info=env.step(np.zeros(3))
            assert not term and not trunc
            expected=env.sim.platform_motion.sample((env.sim.platform_tick-1)/cfg.physics_hz)
            # Bullet advances the kinematic body by one tick after our pose/velocity reset.
            assert np.allclose(env.platform_velocity,expected[1],atol=cfg.platform_accel_max/cfg.physics_hz)
            assert np.allclose(p.getBaseVelocity(env.platform_id,physicsClientId=env.client)[0],env.platform_velocity)
            assert np.linalg.norm(env.platform_position-expected[0])<.002
            assert abs(np.arctan2(np.sin(env.platform_yaw-expected[2]),np.cos(env.platform_yaw-expected[2])))<.002
            assert obs['proprio'].shape==(30,) and obs['pixels'].shape==(3,84,84)
        assert np.linalg.norm(env.platform_velocity-initial_velocity)>.005
        physics=dict(seconds=env.elapsed,platform_motion=info['platform_motion'],
                     final_velocity=env.platform_velocity.tolist(),own_observation_dimensions_unchanged=True)
    finally:
        env.close()
    report=dict(status='passed',motion_samples=rows,periodic_position_tangent_second_derivative=True,
        original_straight_motion_preserved=True,seed_reproducible=True,
        curriculum_rejects_incomplete_or_wrong_phase_validation=True,bullet_integration=physics,
        no_optimizer_or_training_called=True)
    out=Path('runs/platform_motion_checks'); out.mkdir(parents=True,exist_ok=True)
    (out/'checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
