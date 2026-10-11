"""Estimated-state terminal action constraints, separate from the learned policy.

Only self pose/velocity and the RGB tracker enter this controller. No platform
truth, segmentation or teacher actions. Requested policy actions stay in replay;
executed commands are recorded separately, and enter own-command memory.
"""
import numpy as np
import pybullet as p


def bounded_vector(vector,limit):
    vector=np.asarray(vector,dtype=float)
    return vector*min(1.,limit/max(float(np.linalg.norm(vector)),1e-12))


class NearLandingGuard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active=False; self.blend=0.; self.unknown_seconds=0.
        self.last_output=np.zeros(3); self.last_velocity=np.zeros(3)
        self.last_height=None

    def apply(self,requested,state,center,dt,config):
        requested=np.asarray(requested,dtype=float)
        reliable=center.get('position') is not None
        if reliable:
            height=float(state[2]-center['position'][2])
            self.last_height=height
            self.last_velocity=np.asarray(center['velocity'],dtype=float).copy()
            self.last_velocity[2]=0.
            if center.get('near_field'):
                self.active=True
            elif self.active and height>config.near_exit_height:
                # Recover the full command envelope gradually on a deliberate climb.
                self.active=False
        self.blend=float(np.clip(self.blend+(dt/.8 if self.active else -dt/.8),0.,1.))
        if not self.active and self.blend<=0.:
            self.last_output=requested.copy()
            return requested,dict(active=False,blend=0.,unknown_seconds=0.,descent_allowed=True)
        self.unknown_seconds=0. if reliable else self.unknown_seconds+dt
        height=self.last_height if self.last_height is not None else config.near_enter_height
        x=float(np.clip((height-.10)/.70,0.,1.))
        correction_limit=.08+.22*x*x*(3.-2.*x)
        platform_velocity=self.last_velocity[:2]*np.exp(-self.unknown_seconds/1.5)
        relative_command=bounded_vector(requested[:2]-platform_velocity,correction_limit)
        confidence=float(center.get('confidence',0.))
        if reliable:
            error=np.asarray(center['position'])[:2]-state[:2]
            relative_velocity=np.asarray(state[10:12])-platform_velocity
            feedback=bounded_vector(1.2*error-.5*relative_velocity,correction_limit)
            weight=(.85 if center['mode']=='predicted' else .60)*float(np.clip(confidence/.25,0.,1.))
            relative_command=(1.-weight)*relative_command+weight*feedback
        target=np.r_[platform_velocity+relative_command,requested[2]]
        distance=float('inf') if not reliable else float(np.linalg.norm(np.asarray(center['position'])[:2]-state[:2]))
        relative_speed=float(np.linalg.norm(np.asarray(state[10:12])-platform_velocity))
        tilt=max(abs(v) for v in p.getEulerFromQuaternion(state[3:7])[:2])
        descent_allowed=bool(reliable and confidence>=.10 and distance<.25 and relative_speed<.18 and tilt<np.deg2rad(15))
        descent_limit=.08+.04*x
        target[2]=np.clip(target[2],-descent_limit if descent_allowed else 0.,.10)
        if not reliable:
            target[:2]=platform_velocity  # Bounded fallback; no new blind position correction.
        desired=(1.-self.blend)*requested+self.blend*target
        xy_limit=((1.-self.blend)*3.+self.blend*config.near_command_slew_xy)*dt
        z_limit=((1.-self.blend)*1.+self.blend*config.near_command_slew_z)*dt
        output=self.last_output.copy()
        output[:2]+=bounded_vector(desired[:2]-output[:2],xy_limit)
        output[2]+=np.clip(desired[2]-output[2],-z_limit,z_limit)
        self.last_output=output.copy()
        return output,dict(active=self.active,blend=self.blend,unknown_seconds=self.unknown_seconds,
            estimated_height_m=height,relative_correction_limit_m_s=correction_limit,
            descent_allowed=descent_allowed,relative_speed_m_s=relative_speed)


def near_center_reward(state,center):
    if not center.get('near_field') or center.get('position') is None:
        return 0.
    distance=np.linalg.norm(np.asarray(center['position'])[:2]-state[:2])
    relative_velocity=np.asarray(state[10:12])-np.asarray(center['velocity'])[:2]
    return float(center['confidence']*(.025*np.exp(-(distance/.12)**2)-.12*np.sum(relative_velocity**2)))
