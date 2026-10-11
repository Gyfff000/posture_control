"""Near-field checks without optimization, training or checkpoint modification."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pybullet as p

from .center_marker import CenterTracker
from .environment import LIMITS,MovingPlatformLanding
from .near_landing import NearLandingGuard,near_center_reward
from .smooth import smooth_config


def main():
    config=smooth_config(1,True,True)
    out=Path('runs/near_landing_checks'); out.mkdir(exist_ok=True)
    result={}
    tracker=CenterTracker()
    own=np.zeros(20); own[6]=1.; own[2]=1.15
    rgb=np.full((84,84,3),180,np.uint8)
    # Synthetic measurements isolate the estimator; not performance evidence.
    for index in range(30):
        t=index/30.
        own[0]=.2*t+.10
        measurement=dict(position=np.array([.2*t,0.,.3+.002*np.sin(index)]),confidence=.8,depth=.85)
        with patch('landing_repro.center_marker.detect_center',return_value=measurement):
            center=tracker.update(rgb,own[:3],own[3:7],config,t)
    assert center['near_field'] and center['mode']=='observed'
    plane=center['position'][2]
    with patch('landing_repro.center_marker.detect_center',return_value=None):
        own[0]+=.6
        center=tracker.update(rgb,own[:3],own[3:7],config,t+3.)
        assert center['mode']=='predicted' and center['near_field'],center
        assert np.isclose(center['position'][2],plane) and center['velocity'][2]==0.
        assert center['confidence']>config.near_min_confidence
        valid=center.copy()
        expired=tracker.update(rgb,own[:3],own[3:7],config,t+9.)
        assert expired['mode']=='unknown' and expired['position'] is None
    result.update(near_prediction_valid_after_three_seconds=True,
        confidence_at_three_seconds=valid['confidence'],plane_height_not_extrapolated=True,
        bounded_prediction_still_expires=True)
    guard=NearLandingGuard()
    state=np.zeros(20); state[6]=1.
    state[:3]=np.asarray(valid['position'])+[.10,0,.25]
    state[10:13]=valid['velocity']
    previous=None; max_xy_change=0.
    for index in range(120):
        request=np.array([1.5*(-1.)**index,1.5*(-1.)**(index//3),-.5])
        executed,details=guard.apply(request,state,valid,1./30,config)
        if index>40:
            delta=np.linalg.norm(executed[:2]-previous[:2])
            max_xy_change=max(max_xy_change,delta)
            assert delta<=config.near_command_slew_xy/30+1e-9
            assert abs(executed[2]-previous[2])<=config.near_command_slew_z/30+1e-9
            assert np.linalg.norm(executed[:2]-np.asarray(valid['velocity'])[:2])<=details['relative_correction_limit_m_s']+1e-8
        previous=executed.copy()
    assert details['descent_allowed']
    result.update(large_alternating_requests_constrained=True,max_near_xy_command_change_m_s=max_xy_change,
                  near_relative_correction_limit_m_s=details['relative_correction_limit_m_s'])
    unknown=dict(mode='unknown',position=None,velocity=None,confidence=0.,near_field=True)
    for index in range(70):
        executed,details=guard.apply(np.array([-1.5,1.5,-.5]),state,unknown,1./30,config)
        assert np.linalg.norm(executed[:2]-previous[:2])<=config.near_command_slew_xy/30+1e-9
        assert details['active'] and not details['descent_allowed']
        previous=executed.copy()
    assert executed[2]>=-1e-9
    result['estimate_expiry_does_not_restore_large_actions']=True
    state[:2]=np.asarray(valid['position'])[:2]
    centered=near_center_reward(state,valid)
    state[0]+=.4; offset=near_center_reward(state,valid)
    state[0]-=.4; state[10]+=1.; fast=near_center_reward(state,valid)
    assert centered>offset and centered>fast
    result['centering_and_relative_speed_reward']=True
    env=MovingPlatformLanding(config)
    try:
        measured=dict(position=np.array([0.,0.,.3]),confidence=.8,depth=.8)
        with patch('landing_repro.center_marker.detect_center',return_value=measured):
            obs,_=env.reset(seed=1,options={'initial_xyz':[.1,0.,1.1],'platform_speed':0.})
            for _ in range(12):
                obs,_,term,trunc,info=env.step(np.zeros(3,np.float32))
                assert not term
        assert env.center_estimate['near_field']
        with patch('landing_repro.center_marker.detect_center',return_value=None):
            for index in range(95):
                requested=np.array([(-1.)**index,(-1.)**(index//3),0.],np.float32)
                obs,reward,term,trunc,info=env.step(requested)
                assert np.isfinite(reward) and not term and not trunc
                assert np.allclose(obs['proprio'][21:24],info['executed_normalized_action'])
            assert info['center_estimate']['mode']=='predicted' and info['center_estimate']['age_seconds']>3.
        result.update(headless_control_integration=True,executed_command_memory_correct=True,
            policy_action_differs_from_executed_command=not np.allclose(requested,info['executed_normalized_action']),
            optimizer_or_training_called=False,status='passed')
    finally:
        env.close()
    (out/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
