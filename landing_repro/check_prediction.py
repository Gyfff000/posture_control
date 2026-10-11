"""Rendered partial marker, time confidence, reward and headless dropout checks.

Synthetic camera outages in this check are not claimed as learned performance.
"""
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pybullet as p

from .center_marker import CenterTracker,detect_center,prediction_progress
from .environment import MovingPlatformLanding
from .smooth import smooth_config


def main():
    out=Path('runs/prediction_checks'); out.mkdir(exist_ok=True)
    env=MovingPlatformLanding(smooth_config(1,True))
    results={}
    try:
        obs,_=env.reset(seed=1,options={'initial_xyz':[.2,.1,2.1],'platform_speed':.2})
        own=env.state
        raw=env.rgb.copy()
        assert env.center_estimate['mode']=='observed'
        initial=env.center_estimate.copy()
        partial=cv2.warpAffine(raw,np.array([[1.,0.,42.],[0.,1.,0.]]),(84,84),borderValue=(180,180,180))
        assert detect_center(partial,own[:3],own[3:7],env.cfg) is None
        tracker=CenterTracker()
        assert tracker.update(raw,own[:3],own[3:7],env.cfg,0.)['mode']=='observed'
        first=tracker.update(partial,own[:3],own[3:7],env.cfg,1./30)
        assert first['mode']=='predicted' and np.isclose(first['age_seconds'],1./30)
        results['partial_marker_predicts_on_first_policy_frame']=True
        blank=np.full_like(raw,180)
        confidences=[first['confidence']]
        for t in (.2,.4,.8,1.2):
            state=tracker.update(blank,own[:3],own[3:7],env.cfg,t)
            assert state['mode']=='predicted'
            expected=initial['confidence']*np.exp(-t/.85)
            assert np.isclose(state['confidence'],expected)
            confidences.append(state['confidence'])
        assert np.all(np.diff(confidences)<0)
        unknown=tracker.update(blank,own[:3],own[3:7],env.cfg,2.1)
        assert unknown['mode']=='unknown' and unknown['position'] is None and unknown['confidence']==0.
        recovered=tracker.update(raw,own[:3],own[3:7],env.cfg,2.2)
        assert recovered['mode']=='observed' and recovered['age_seconds']==0.
        results.update(confidence_decays_with_simulation_time=True,confidence_samples=confidences,
                       expired_prediction_hidden=True,reacquisition_restores_confidence=True)
        previous=dict(position=[0,0,0],velocity=[.3,0,0])
        current=dict(mode='predicted',confidence=.6)
        toward=prediction_progress([1,0,1],[.98,0,1],previous,current,1./30)
        away=prediction_progress([1,0,1],[1.02,0,1],previous,current,1./30)
        stationary=prediction_progress([1,0,1],[1,0,1],previous,current,1./30)
        weak=prediction_progress([1,0,1],[.98,0,1],previous,dict(mode='predicted',confidence=.3),1./30)
        assert toward>0 and away<0 and stationary==0. and np.isclose(weak,toward/2)
        results.update(reward_approach=toward,reward_away=away,reward_stationary=stationary,
                       reward_scales_with_confidence=True)
        original_camera=p.getCameraImage
        def camera_partial(*args,**kwargs):
            w,h,rgba,depth,seg=original_camera(*args,**kwargs)
            rgba=np.asarray(rgba).copy(); rgba[...,:3]=partial
            return w,h,rgba,depth,seg
        with patch('landing_repro.environment.p.getCameraImage',side_effect=camera_partial):
            _,_,term,trunc,info=env.step(np.zeros(3,np.float32))
        assert env.visible and info['center_estimate']['mode']=='predicted'
        assert np.isfinite(info['prediction_reward']) and not term
        assert np.array_equal(env._observation()['pixels'][-1],cv2.cvtColor(partial,cv2.COLOR_RGB2GRAY))
        results['partial_marker_reward_branch_even_with_platform_pixels']=True
        def camera_blank(*args,**kwargs):
            w,h,rgba,depth,seg=original_camera(*args,**kwargs)
            rgba=np.asarray(rgba).copy(); rgba[...,:3]=blank
            return w,h,rgba,depth,np.full_like(seg,-1)
        # No GUI calls: the estimator must run every policy step on its own.
        with patch('landing_repro.environment.p.getCameraImage',side_effect=camera_blank):
            frames=0
            while not term and not trunc:
                _,_,term,trunc,info=env.step(np.zeros(3,np.float32)); frames+=1
                assert frames<70
                if frames==31:
                    assert not term and info['center_estimate']['mode']=='predicted'
            assert info['event']=='lost_view' and info['center_estimate']['mode']=='unknown'
        results.update(headless_prediction_every_control_step=True,
                       reliable_prediction_extends_one_second_outage=True,
                       bounded_dropout_termination_seconds=frames/30.,raw_images_have_no_annotation=True)
        with patch('landing_repro.center_marker.detect_center',return_value=None):
            env.reset(seed=2)
            assert env.center_estimate['mode']=='unknown'
            assert env.center_estimate['position'] is None
        results['no_prior_observation_no_truth_fallback']=True
        results['status']='passed'
    finally:
        env.close()
    (out/'checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
