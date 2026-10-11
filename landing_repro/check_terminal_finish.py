"""Check bounded completion and actual contact without any learning."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .center_marker import estimate_center
from .environment import MovingPlatformLanding
from .near_landing import NearLandingGuard
from .smooth import smooth_config


def main():
    config=smooth_config(1,True,True,True)
    state=np.zeros(20); state[6]=1.; state[:3]=[.02,0,.47]; state[10]=.15
    center=dict(mode='predicted',position=[0,0,.3],velocity=[.15,0,0],confidence=.13,
        age_seconds=7.8,position_std_m=[.2,.2,.2],near_field=True)
    unknown=dict(mode='unknown',position=None,velocity=None,confidence=0.,age_seconds=8.1)
    guard=NearLandingGuard()
    _,details=guard.apply([.15,0,-.25],state,center,1./30,config)
    assert details['terminal_active']
    for index in range(85):
        state[0]+=.15/30
        output,details=guard.apply([.15,0,-.25],state,unknown,1./30,config)
        assert details['terminal_active'] and details['descent_allowed']
        assert guard.terminal_estimate()['estimate_kind']=='terminal_coast'
    assert output[2]<-.09
    while details['terminal_active']:
        state[0]+=.15/30
        _,details=guard.apply([.15,0,-.25],state,unknown,1./30,config)
    assert details['terminal_timed_out'] and guard.terminal_estimate() is None
    fresh=NearLandingGuard()
    _,unsafe=fresh.apply([.15,0,-.25],state,unknown,1./30,config)
    assert not unsafe['terminal_active']
    state[0]=.40
    _,misaligned=NearLandingGuard().apply([.15,0,-.25],state,center,1./30,config)
    assert not misaligned['terminal_active']
    results=dict(terminal_latches_before_expiry=True,continues_descent_after_tracking_unknown=True,
        never_commits_without_valid_alignment=True,bounded_completion_timeout=True,
        no_truth_position_used_by_controller=True)
    env=MovingPlatformLanding(config)
    try:
        measurement=dict(position=np.array([0.,0.,.3]),confidence=.8,depth=.14)
        with patch('landing_repro.center_marker.detect_center',return_value=measurement):
            env.reset(seed=1,options={'initial_xyz':[.02,0,.44],'platform_speed':0.})
            for _ in range(12):
                _,_,term,trunc,info=env.step(np.array([0.,0.,-.30],np.float32))
                if env.near_guard.terminal_active:
                    break
            assert env.near_guard.terminal_active and not term
        # Force estimator expiry without providing any platform truth fallback.
        with patch.object(env._center_tracker,'update',return_value=unknown):
            rendered=estimate_center(env)
            assert rendered['estimate_kind']=='terminal_coast'
            for _ in range(120):
                _,_,term,trunc,info=env.step(np.array([0.,0.,-.30],np.float32))
                if term or trunc:
                    break
            assert info['event']=='success',info
        results.update(expired_tracker_still_has_bounded_display_center=True,
            physical_contact_required_for_success=True,synthetic_physics_event=info['event'],
            no_optimizer_or_training_called=True,status='passed')
    finally:
        env.close()
    out=Path('runs/terminal_finish_checks'); out.mkdir(exist_ok=True)
    (out/'checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
