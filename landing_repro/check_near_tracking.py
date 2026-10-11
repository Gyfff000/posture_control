"""Targeted checks: partial RGB flow, bounded loss and variable-motion finish."""
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from .center_marker import CenterTracker,estimate_center
from .environment import MovingPlatformLanding
from .motion_curriculum import motion_config
from .near_landing import NearLandingGuard
from .partial_marker import PartialMarkerTracker
from .smooth import smooth_config


def main():
    cfg=motion_config(smooth_config,1,True,True,True)
    image=np.full((84,84,3),255,np.uint8)
    rng=np.random.default_rng(81)
    for x,y in rng.integers(20,65,(40,2)):
        cv2.rectangle(image,(int(x),int(y)),(int(x+2),int(y+2)),(0,0,0),-1)
    track=PartialMarkerTracker()
    track.seed(image,dict(quad=np.array([[15,15],[69,15],[69,69],[15,69]],np.float32),center_pixel=[42,42]))
    own=np.array([0.,0.,1.3]); quat=[0,0,0,1]
    errors=[]
    for index in range(1,31):
        matrix=np.array([[1,0,index*.75],[0,1,index*.20]],np.float32)
        moved=cv2.warpAffine(image,matrix,(84,84),borderValue=(255,255,255))
        measurement=track.update(moved,own,quat,.3,cfg)
        assert measurement is not None,index
        expected=np.array([cfg.camera_offset_x-index*.20/42*.992,-index*.75/42*.992,.3])
        errors.append(float(np.linalg.norm(measurement['position']-expected)))
    assert max(errors)<.02,errors
    assert track.update(np.zeros_like(image),own,quat,.3,cfg) is None
    # A partial correction must not rejuvenate the age of the last full marker.
    tracker=CenterTracker()
    full=dict(position=np.array([0.,0.,.3]),confidence=.8,depth=.5)
    with patch('landing_repro.center_marker.detect_center',return_value=full):
        for i in range(30): tracker.update(image,[0,0,.8],quat,cfg,i/30)
    with patch('landing_repro.center_marker.detect_center',return_value=None):
        with patch.object(tracker.partial_tracker,'update',return_value=dict(position=np.array([0.,0.,.3]))):
            for i in range(30,100): result=tracker.update(image,[0,0,.8],quat,cfg,i/30)
    assert result['estimate_kind']=='partial_flow' and result['age_seconds']>2,result
    assert result['measurement_age_seconds']==0 and result['confidence']<.8
    with patch('landing_repro.center_marker.detect_center',return_value=None):
        for i in range(100,330): result=tracker.update(np.zeros_like(image),[0,0,.8],quat,cfg,i/30)
    assert result['mode']=='unknown'
    # Reported failure: at ~10 cm, old variable entry gate (.10) was too late
    # because estimated plane can be 3 cm below its physical surface.
    state=np.zeros(20); state[6]=1.; state[:3]=[.02,0,.45]; state[10]=.15
    center=dict(mode='predicted',position=[0,0,.27],velocity=[.15,0,0],confidence=.25,
        age_seconds=3.,position_std_m=[.04,.04,.02],velocity_std_m_s=[.02,.02,.02],near_field=True)
    unknown=dict(mode='unknown',position=None,velocity=None,confidence=0.,age_seconds=8.1)
    guard=NearLandingGuard()
    _,details=guard.apply([.15,0,-.2],state,center,1/30,cfg)
    assert details['terminal_active'],details
    for _ in range(20):
        state[0]+=.15/30; state[2]-=.10/30
        output,details=guard.apply([.15,0,-.2],state,unknown,1/30,cfg)
        assert details['descent_allowed'] and details['terminal_active']
    assert output[2]<0 and guard.terminal_estimate() is not None
    # An accelerating platform plus large uncertainty must still block commitment.
    unsafe=dict(center,position_std_m=[.29,.29,.02],velocity_std_m_s=[.2,.2,.02])
    _,details=NearLandingGuard().apply([.15,0,-.2],state,unsafe,1/30,cfg)
    assert not details['terminal_active']
    # Refresh terminal state from an actual visual update, never mere prediction.
    fresh=dict(center,position=[state[0],0,.27],measurement_age_seconds=0.,estimate_kind='partial_flow')
    _,details=guard.apply([.15,0,-.2],state,fresh,1/30,cfg)
    assert details['terminal_refreshes']==1 and guard.terminal_blind_seconds==0
    env=MovingPlatformLanding(cfg)
    try:
        full=dict(position=np.array([0.,0.,.3]),confidence=.8,depth=.18)
        with patch('landing_repro.center_marker.detect_center',return_value=full):
            env.reset(seed=81,options={'initial_xyz':[.02,0,.48],'platform_speed':0.})
            for _ in range(15):
                _,_,term,trunc,info=env.step(np.array([0,0,-.3],np.float32))
                if env.near_guard.terminal_active: break
            assert env.near_guard.terminal_active and not term
        with patch.object(env._center_tracker,'update',return_value=unknown):
            assert estimate_center(env)['estimate_kind']=='terminal_coast'
            for _ in range(90):
                _,_,term,trunc,info=env.step(np.array([0,0,-.3],np.float32))
                if term or trunc: break
            assert info['event']=='success',info['event']
    finally: env.close()
    result=dict(status='passed',optical_flow_max_world_error_m=max(errors),
        clipped_marker_flow_checked=True,rejects_blank_camera=True,partial_update_does_not_reset_full_age=True,
        confidence_decays_and_missing_view_expires=True,
        earlier_variable_motion_completion=True,uncertain_commitment_rejected=True,
        fresh_visual_update_refreshes_finish=True,physical_contact_after_loss=info['event'],
        optimizer_updates=0)
    out=Path('runs/near_tracking_checks'); out.mkdir(exist_ok=True)
    (out/'checks.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
