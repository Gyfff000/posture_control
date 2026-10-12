"""Camera center estimation and bounded, confidence-weighted prediction.

No segmentation IDs or platform truth are used. This is an auxiliary estimator,
not an explanation of the network's hidden features or a policy input.
"""
from functools import lru_cache
from collections import deque

import cv2
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw


@lru_cache(maxsize=1)
def reference_mask():
    # Known printed design and physical size are calibration inputs, not live truth.
    from .environment import make_assets
    _, path = make_assets()
    gray = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    return cv2.resize(gray,(96,96))[8:88,8:88] < 100


def detect_center(rgb, own_position, own_quaternion, config):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape
    dark=(gray<100).astype(np.uint8)*255
    # TinyRenderer undersamples the thin black square into separated dots at 84px.
    # Close one-pixel gaps; keep original contours as candidates for close views.
    joined=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(joined,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    raw_contours,_=cv2.findContours(dark,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours=list(contours)+list(raw_contours)
    regions,_=cv2.findContours(cv2.dilate(dark,np.ones((3,3),np.uint8)),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for region in regions:
        x,y,rw,rh=cv2.boundingRect(region)
        ys,xs=np.nonzero(dark[y:y+rh,x:x+rw])
        if len(xs)>=12:
            points=np.column_stack([xs+x,ys+y]).astype(np.int32).reshape(-1,1,2)
            contours.append(cv2.convexHull(points))
    for contour in sorted(contours,key=cv2.contourArea,reverse=True):
        area=cv2.contourArea(contour)
        if area<25 or area>.95*h*w:
            continue
        quad=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
        if len(quad)!=4 or not cv2.isContourConvex(quad):
            continue
        quad=quad[:,0].astype(np.float32)
        if np.any(quad<1) or np.any(quad[:,0]>w-2) or np.any(quad[:,1]>h-2):
            continue  # Clipped shapes cannot give a trustworthy complete marker pose.
        center=quad.mean(axis=0)
        quad=quad[np.argsort(np.arctan2(quad[:,1]-center[1],quad[:,0]-center[0]))]
        quad=np.roll(quad,-np.argmin(quad.sum(axis=1)),axis=0)
        warp=cv2.warpPerspective(gray,cv2.getPerspectiveTransform(quad,np.array([[0,0],[95,0],[95,95],[0,95]],np.float32)),(96,96))
        low,high=np.percentile(warp,[5,95])
        if high-low<50:
            continue
        dark=warp[8:88,8:88]<(low+.45*(high-low))
        ref=reference_mask()
        overlap=max(np.logical_and(dark,np.rot90(ref,k)).sum()/max(1,np.logical_or(dark,np.rot90(ref,k)).sum()) for k in range(4))
        if overlap<.35:
            continue
        f=h/(2*np.tan(np.deg2rad(config.camera_fov)/2))
        intrinsic=np.array([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float64)
        obj=np.array([[-.5,.5,0],[.5,.5,0],[.5,-.5,0],[-.5,-.5,0]],np.float64)
        ok,rvec,tvec=cv2.solvePnP(obj,quad.astype(np.float64),intrinsic,np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(tvec).all() or not .03<tvec[2,0]<100:
            continue
        projected,_=cv2.projectPoints(obj,rvec,tvec,intrinsic,np.zeros(5))
        reprojection=float(np.linalg.norm(projected[:,0]-quad,axis=1).mean())
        if reprojection>2.:
            continue
        rotation=np.array(p.getMatrixFromQuaternion(own_quaternion)).reshape(3,3)
        forward=rotation@np.array([0.,0.,-1.])
        up=rotation@np.array([1.,0.,0.])
        right=np.cross(forward,up)
        eye=own_position+rotation@np.array([config.camera_offset_x,0.,config.camera_offset_z])
        world=eye+np.column_stack([right,-up,forward])@tvec[:,0]
        return dict(position=world,confidence=float(overlap),reprojection_pixels=reprojection,
                    depth=float(tvec[2,0]),quad=quad,
                    center_pixel=(intrinsic@tvec[:,0])[:2]/tvec[2,0])
    return None


class CenterTracker:
    def __init__(self,max_prediction_seconds=2.,confidence_tau=.85,min_confidence=.10):
        self.max_age=max_prediction_seconds
        self.confidence_tau=confidence_tau
        self.min_confidence=min_confidence
        self.reset()

    def reset(self):
        self.x=None; self.cov=None; self.time=None; self.last_seen=None
        self.last_quality=0.
        self.observations=0; self.near_field=False
        self.plane_samples=deque(maxlen=12)
        from .partial_marker import PartialMarkerTracker
        self.partial_tracker=PartialMarkerTracker()
        self.last_visual_update=None; self.partial_updates=0
        self.result=dict(mode='unknown',position=None,velocity=None,age_seconds=None,confidence=0.)
        self.line_id=-1; self.text_id=-1

    def update(self,rgb,own_position,own_quaternion,config,time_s):
        if self.time is not None and time_s<self.time-1e-9:
            self.reset()
        if self.time is not None and abs(time_s-self.time)<1e-9:
            return self.result
        dt=0. if self.time is None else time_s-self.time
        self.time=time_s
        if self.x is not None:
            transition=np.eye(6); transition[:3,3:]=np.eye(3)*dt
            self.x=transition@self.x
            # Continuous white-acceleration covariance is independent of frame batching.
            # Variable-motion tasks increase uncertainty using a configured prior,
            # not the platform's true acceleration or future trajectory.
            process_scale=.03 if self.near_field else .20
            if config.platform_motion!='constant':
                process_scale=max(process_scale,config.platform_accel_max)
            density=process_scale**2
            eye=np.eye(3)
            process=density*np.block([[eye*dt**3/3,eye*dt**2/2],[eye*dt**2/2,eye*dt]])
            self.cov=transition@self.cov@transition.T+process
        measurement=detect_center(rgb,own_position,own_quaternion,config)
        accepted=False
        reason='not_detected_or_incomplete'
        if measurement is not None and measurement['confidence']<config.center_min_measurement_quality:
            reason='low_measurement_quality'
            measurement=None
        if measurement is not None:
            reason='innovation_rejected'
            z=measurement['position']
            reacquire_age=config.near_prediction_seconds if self.near_field else self.max_age
            if self.x is None or time_s-self.last_seen>reacquire_age:
                self.x=np.r_[z,np.zeros(3)]
                self.cov=np.diag([.03**2]*3+[.7**2]*3)
                accepted=True
            else:
                noise=(.015+.01*measurement['depth'])**2*np.eye(3)
                innovation=z-self.x[:3]
                s=self.cov[:3,:3]+noise
                if innovation@np.linalg.solve(s,innovation)<25:
                    gain=self.cov[:,:3]@np.linalg.inv(s)
                    self.x+=gain@innovation
                    # Joseph update keeps covariance positive under repeated updates.
                    kh=np.zeros((6,6)); kh[:,:3]=gain
                    self.cov=(np.eye(6)-kh)@self.cov@(np.eye(6)-kh).T+gain@noise@gain.T
                    speed=np.linalg.norm(self.x[3:])
                    if speed>3.:
                        self.x[3:]*=3./speed
                    accepted=True
            if accepted:
                self.last_seen=time_s
                self.last_visual_update=time_s
                self.last_quality=measurement['confidence']
                self.observations+=1
                self.plane_samples.append(float(self.x[2]))
                reason='complete_marker_accepted'
                if config.near_partial_tracking:
                    self.partial_tracker.seed(rgb,measurement)
        partial_accepted=False
        if (not accepted and self.near_field and config.near_partial_tracking
                and self.last_seen is not None and time_s-self.last_seen<=config.near_prediction_seconds):
            partial=self.partial_tracker.update(rgb,own_position,own_quaternion,
                float(np.median(self.plane_samples)),config)
            if partial is not None:
                # Weaker than full-marker PnP. A partial update never renews full age.
                noise=np.eye(3)*(.04+.005*(time_s-self.last_seen))**2
                innovation=partial['position']-self.x[:3]
                s=self.cov[:3,:3]+noise
                if innovation@np.linalg.solve(s,innovation)<16 and np.linalg.norm(innovation[:2])<.25:
                    gain=self.cov[:,:3]@np.linalg.inv(s)
                    self.x+=gain@innovation
                    kh=np.zeros((6,6)); kh[:,:3]=gain
                    self.cov=(np.eye(6)-kh)@self.cov@(np.eye(6)-kh).T+gain@noise@gain.T
                    self.last_visual_update=time_s; self.partial_updates+=1
                    partial_accepted=True; reason='partial_marker_flow_accepted'
        age=None if self.last_seen is None else max(0.,time_s-self.last_seen)
        std=None if self.cov is None else np.sqrt(np.maximum(0.,np.diag(self.cov)[:3]))
        if self.x is not None and config.near_landing_assist:
            height=float(own_position[2]-self.x[2])
            distance=float(np.linalg.norm(np.asarray(own_position)[:2]-self.x[:2]))
            # Arm before the 1m square fills the camera. Never arm from stale data.
            if (not self.near_field and self.observations>=5 and age<=.5
                    and -.05<height<=config.near_enter_height and distance<=.65
                    and max(std[:2])<.15 and self.last_quality*np.exp(-age/self.confidence_tau)>=.25):
                self.near_field=True
            elif self.near_field and (height>config.near_exit_height or distance>1.3):
                self.near_field=False
            if self.near_field:
                # Horizontal moving platform assumption; do not extrapolate noisy PnP vz.
                self.x[2]=float(np.median(self.plane_samples)); self.x[5]=0.
        tau=config.near_confidence_tau if self.near_field else self.confidence_tau
        maximum=config.near_prediction_seconds if self.near_field else self.max_age
        minimum=config.near_min_confidence if self.near_field else self.min_confidence
        confidence=0. if age is None else float(self.last_quality*np.exp(-age/tau))
        if self.near_field and not accepted:
            confidence*=float(np.exp(-(max(std[:2])/.5)**2))
        valid=self.x is not None and age<=maximum and confidence>=minimum
        if self.near_field and max(std[:2])>config.near_max_position_std:
            valid=False
        self.result=dict(mode='observed' if accepted else 'predicted' if valid else 'unknown',
                         position=self.x[:3].tolist() if valid else None,
                         velocity=self.x[3:].tolist() if valid else None,age_seconds=age,
                         confidence=confidence if valid else 0.,
                         near_field=self.near_field,accepted_observations=self.observations,
                         last_measurement_quality=self.last_quality,
                         estimate_kind='partial_flow' if partial_accepted else 'full_marker' if accepted else 'kalman_prediction',
                         measurement_age_seconds=None if self.last_visual_update is None else time_s-self.last_visual_update,
                         partial_updates=self.partial_updates,
                         velocity_std_m_s=None if self.cov is None else np.sqrt(np.maximum(0.,np.diag(self.cov)[3:])).tolist(),
                         position_std_m=None if std is None else std.tolist(),
                         reason=reason if valid or age is None else 'prediction_expired_or_low_confidence',
                         source='auxiliary RGB PnP + optional partial-marker optical flow + own pose + Kalman filter; heuristic confidence, not network output')
        return self.result


def estimate_center(env):
    if not hasattr(env,'_center_tracker'):
        env._center_tracker=CenterTracker(env.cfg.center_prediction_seconds,
            env.cfg.center_confidence_tau,env.cfg.center_min_confidence)
    tracker=env._center_tracker
    state=env.state
    rgb=getattr(env,'center_rgb',env.rgb)
    # Optical flow must not pair frames with different resolutions.
    if tracker.partial_tracker.gray is not None and tracker.partial_tracker.gray.shape!=rgb.shape[:2]:
        from .partial_marker import PartialMarkerTracker
        tracker.partial_tracker=PartialMarkerTracker()
    result=tracker.update(rgb,state[:3],state[3:7],env.cfg,env.elapsed)
    if env.cfg.terminal_finish_enabled and result['position'] is None and hasattr(env,'near_guard'):
        terminal=env.near_guard.terminal_estimate()
        if terminal is not None:
            terminal['tracker_reason']=result.get('reason')
            return terminal
    return result


def prediction_progress(before_position,after_position,previous_estimate,current_estimate,dt,weight=4.):
    """Score actual horizontal motion toward ONE fixed extrapolated target.

    Both distances use the same target: target motion and measurement jumps
    cannot earn progress for a stationary drone. No simulator platform truth.
    """
    if current_estimate['mode']!='predicted' or previous_estimate.get('position') is None:
        return 0.
    target=np.asarray(previous_estimate['position'])[:2]+dt*np.asarray(previous_estimate['velocity'])[:2]
    before=np.linalg.norm(np.asarray(before_position)[:2]-target)
    after=np.linalg.norm(np.asarray(after_position)[:2]-target)
    return float(weight*current_estimate['confidence']*np.clip(before-after,-.10,.10))


def update_center_marker(env):
    # Rendering must not assimilate the same camera frame a second time.
    # _observe() already advanced the estimator exactly once for this step.
    # Older display-only configurations need one cached estimate per RGB frame.
    if not env.cfg.prediction_guidance and getattr(env,'_display_estimate_frame',None) is not env.rgb:
        env.center_estimate=estimate_center(env)
        env._display_estimate_frame=env.rgb
    result=env.center_estimate
    tracker=env._center_tracker
    if env.gui:
        if result['position'] is None:
            for identifier in (tracker.line_id,tracker.text_id):
                if identifier>=0:
                    p.removeUserDebugItem(identifier,physicsClientId=env.client)
            tracker.line_id=tracker.text_id=-1
        else:
            base=np.asarray(result['position'])
            top=base+[0,0,1.]
            color=[.10,.90,.25] if result['mode']=='observed' else [1.,.55,.0]
            tracker.line_id=p.addUserDebugLine(base,top,color,4,replaceItemUniqueId=tracker.line_id,physicsClientId=env.client)
            label=('VISION center' if result['mode']=='observed' else ('PARTIAL FLOW' if result.get('estimate_kind')=='partial_flow' else 'FINISH COAST' if result.get('estimate_kind')=='terminal_coast' else 'NEAR PREDICTED' if result.get('near_field') else 'PREDICTED')+f' {result["age_seconds"]:.1f}s')+f' C={result["confidence"]:.2f}'
            tracker.text_id=p.addUserDebugText(label,top,textColorRGB=color,textSize=1.1,
                replaceItemUniqueId=tracker.text_id,physicsClientId=env.client)
    return result


def clipped_segment(a,b):
    # Clip homogeneous points against the complete OpenGL view volume.
    lo,hi=0.,1.
    for axis in range(3):
        for sign in (-1.,1.):
            fa=a[3]+sign*a[axis]; fb=b[3]+sign*b[axis]
            if fa<0 and fb<0:
                return None
            if fa<0 or fb<0:
                crossing=fa/(fa-fb)
                if fa<0: lo=max(lo,crossing)
                else: hi=min(hi,crossing)
    if lo>hi:
        return None
    return a+lo*(b-a),a+hi*(b-a)


def overlay(image,result,view,projection):
    image=image.copy()
    draw=ImageDraw.Draw(image)
    mode=result['mode']
    color=(20,220,70) if mode=='observed' else (255,145,0)
    label='VISION estimate' if mode=='observed' else ('PARTIAL FLOW' if result.get('estimate_kind')=='partial_flow' else 'FINISH COAST' if result.get('estimate_kind')=='terminal_coast' else 'NEAR PRED' if result.get('near_field') else 'PREDICTED')+f' {result["age_seconds"]:.1f}s' if mode=='predicted' else 'CENTER UNKNOWN'
    if mode!='unknown':
        label+=f' C={result["confidence"]:.2f}'
    if result['position'] is not None:
        base=np.r_[result['position'],1.]
        top=base+np.array([0,0,1.,0])
        matrix=np.asarray(projection).reshape(4,4,order='F')@np.asarray(view).reshape(4,4,order='F')
        line=clipped_segment(matrix@base,matrix@top)
        def pixel(v):
            return ((v[0]/v[3]+1)*image.width/2,(1-v[1]/v[3])*image.height/2)
        if line is not None and min(line[0][3],line[1][3])>1e-8:
            draw.line([pixel(line[0]),pixel(line[1])],fill=color,width=3)
        point=matrix@base
        if point[3]>0 and np.all(np.abs(point[:3])<=point[3]):
            x,y=pixel(point)
            draw.line([(x-5,y),(x+5,y)],fill=color,width=2)
            draw.line([(x,y-5),(x,y+5)],fill=color,width=2)
    draw.rectangle((0,0,image.width,16),fill=(25,25,25))
    draw.text((4,2),label,fill=color if mode!='unknown' else (190,190,190))
    return image
