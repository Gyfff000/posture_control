from pathlib import Path
import json
from dataclasses import asdict
from landing_repro.smooth import smooth_config
from landing_repro.motion_curriculum import motion_config
from landing_repro.height_curriculum import HEIGHT_LEVELS,height_config,height_promotion_ready
from landing_repro.environment import MovingPlatformLanding

base=motion_config(smooth_config,2,True,True,True)
checked=[]
for level,h in enumerate(HEIGHT_LEVELS):
    cfg=height_config(base,level)
    changes={k for k,v in asdict(cfg).items() if v!=asdict(base)[k]}
    assert changes <= {'initial_height','offset_z','episode_seconds','center_camera_size'}, changes
    assert cfg.near_partial_tracking and cfg.near_controller_version==2
    env=MovingPlatformLanding(cfg,gui=False)
    try:
        for seed in (301,302):
            obs,_=env.reset(seed=seed)
            assert obs['pixels'].shape==(3,84,84)
            actual=float(env.state[2]-cfg.platform_top)
            assert abs(actual-h)<=.300001,(level,actual,h)
        checked.append({'level':level,'height_m':h,'actual_height_m':actual,
                        'episode_seconds':cfg.episode_seconds,'auxiliary_cap':cfg.center_camera_size})
    finally:
        env.close()

def row(level=1,phase=2,successes=17,soft=16,n=20):
    return dict(height_level=level,motion_phase=phase,episodes=n,successes=successes,soft_landings=soft)
assert not height_promotion_ready([row(),row()],2,1)
assert height_promotion_ready([row(),row(successes=16),row()],2,1)
assert not height_promotion_ready([row(),row(),row(successes=15)],2,1)
assert not height_promotion_ready([row(),row(),row(successes=19)],2,1)
assert not height_promotion_ready([row(),row(),row(soft=11)],2,1)
assert not height_promotion_ready([row(),row(),row(level=2)],2,1)
assert not height_promotion_ready([row(),row(),row(phase=3)],2,1)
assert not height_promotion_ready([row(n=5)]*3,2,1,5)
result={'status':'passed','height_resets':checked,'promotion_checks':8}
Path('runs/height_curriculum_checks').mkdir(parents=True,exist_ok=True)
Path('runs/height_curriculum_checks/height_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
