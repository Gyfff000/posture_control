"""Real camera calibration checks, not learned landing performance."""
from pathlib import Path
from dataclasses import replace
import json,numpy as np,pybullet as p
from .environment import MovingPlatformLanding
from .smooth import smooth_config
from .motion_curriculum import motion_config
from .height_curriculum import HEIGHT_LEVELS,height_config
rows=[]
for height in (3.,3.5,4.,5.,6.,8.,10.,12.,15.):
    cfg=height_config(motion_config(smooth_config,2,True,True,True),HEIGHT_LEVELS.index(height))
    env=MovingPlatformLanding(cfg)
    try:
        for seed in (301,302,303):
            obs,_=env.reset(seed=seed)
            assert obs['pixels'].shape==(3,84,84) and env.rgb.shape==(84,84,3)
            center=env.center_estimate
            assert center['mode']=='observed',(height,seed,center)
            error=float(np.linalg.norm(np.asarray(center['position'])-(env.platform_position+[0,0,.04])))
            assert error<1.,(height,seed,error)  # Loose initialization check; not a landing tolerance.
            rows.append(dict(height_m=height,seed=seed,auxiliary_size=env.center_rgb.shape[0],
                center_error_m=error,confidence=center['confidence']))
        if height==15.:
            for test_height in (2.7,2.3,3.1):
                own=env.state
                p.resetBasePositionAndOrientation(env.drone,[own[0],own[1],cfg.platform_top+test_height],own[3:7],physicsClientId=env.client)
                env.elapsed+=1./cfg.policy_hz;env._observe()
                assert env._observation()['pixels'].shape==(3,84,84)
            assert cfg.episode_seconds>=82.
    finally:env.close()
# Current two-meter stage retains the exact original rendering and horizon.
cfg=height_config(motion_config(smooth_config,2,True,True,True),1)
assert cfg.center_camera_size==84 and cfg.episode_seconds==60
out=Path('runs/high_altitude_camera_checks');out.mkdir(exist_ok=True)
result=dict(status='passed',note='Initialization/calibration and resolution-switch checks only; not landing success rates',rows=rows,actor_shape=[3,84,84],resolution_switches=3)
(out/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
