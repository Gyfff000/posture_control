"""Actual rendered-camera checks; synthetic blank frames test dropout handling."""
import json
from pathlib import Path
import numpy as np
from .center_marker import CenterTracker,detect_center,overlay,update_center_marker
from .environment import MovingPlatformLanding
from .scene_views import inspection_views,save_inspection
from .smooth import smooth_config
import pybullet as p


def main():
    out=Path('runs/center_marker_checks'); out.mkdir(exist_ok=True)
    env=MovingPlatformLanding(smooth_config(1))
    results={}
    try:
        obs,_=env.reset(seed=1,options={'initial_xyz':[.2,.1,2.1],'platform_speed':.2,'platform_yaw':.4})
        raw=env.rgb.copy()
        from PIL import Image
        Image.fromarray(raw).resize((672,672)).save(out/'raw_camera.png')
        own=env.state
        measurement=detect_center(raw,own[:3],own[3:7],env.cfg)
        assert measurement is not None, 'No full marker detected in rendered camera'
        truth=env.platform_position+np.array([0,0,.04])  # Truth used ONLY to score this check.
        error=np.linalg.norm(measurement['position']-truth)
        assert error<.20,(measurement['position'],truth,error)
        results['rendered_camera_center_error_m']=float(error)
        status=update_center_marker(env)
        assert status['mode']=='observed'
        assert np.array_equal(env.rgb,raw) and np.array_equal(obs['pixels'],env._observation()['pixels'])
        save_inspection(inspection_views(env),out/'observed.png','GREEN: camera-only square-marker estimate; display annotations do not enter the policy')
        tracker=env._center_tracker
        for index in range(12):
            env.step(np.zeros(3,np.float32))
            update_center_marker(env)
        assert tracker.result['mode']=='observed'
        blank=np.full_like(env.rgb,180)
        t=env.elapsed
        before=np.array(tracker.result['position'])
        velocity=np.array(tracker.result['velocity'])
        predicted=tracker.update(blank,env.state[:3],env.state[3:7],env.cfg,t+.4)
        assert predicted['mode']=='predicted' and np.isclose(predicted['age_seconds'],.4)
        assert np.allclose(predicted['position'],before+.4*velocity)
        # Show artificial dropout only in this isolated check, never in live training.
        from .scene_views import close_view
        from PIL import Image
        camera_projection=p.computeProjectionMatrixFOV(env.cfg.camera_fov,1.,env.cfg.camera_near,100.)
        views=(close_view(env,env.state[:3],.32,-25),close_view(env,env.platform_position+[0,0,.04],1.65,-55),
               overlay(Image.fromarray(blank).resize((250,250)),predicted,env.camera_view,camera_projection))
        save_inspection(views,out/'predicted.png','ORANGE: isolated synthetic 0.4 s camera dropout check, not a training result')
        expired=tracker.update(blank,env.state[:3],env.state[3:7],env.cfg,t+2.1)
        assert expired['mode']=='unknown' and expired['position'] is None
        reset=tracker.update(blank,env.state[:3],env.state[3:7],env.cfg,0.)
        assert reset['mode']=='unknown' and tracker.x is None
        results.update(dropout_constant_velocity_prediction=True,timeout_hides_unreliable_prediction=True,
                       episode_reset_clears_estimate=True,raw_model_images_unchanged=True,status='passed')
    finally:
        env.close()
    (out/'checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
