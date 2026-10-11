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
        self.last_plane=None
        self.terminal_active=False; self.terminal_attempted=False; self.terminal_seconds=0.
        self.terminal_position=None; self.terminal_velocity=None
        self.terminal_quality=0.; self.terminal_age=0.
        self.terminal_duration=0.
        self.terminal_blind_seconds=0.; self.terminal_refreshes=0
        self.terminal_position_std=0.; self.terminal_velocity_std=0.

    def completion_radius(self,state,config,position=None,position_std=None,velocity_std=None,blind_seconds=None):
        position=self.terminal_position if position is None else np.asarray(position)
        position_std=self.terminal_position_std if position_std is None else position_std
        velocity_std=self.terminal_velocity_std if velocity_std is None else velocity_std
        blind_seconds=self.terminal_blind_seconds if blind_seconds is None else blind_seconds
        remaining=max(0.,float(state[2]-position[2])-.035)/config.terminal_finish_speed+.25
        horizon=blind_seconds+remaining
        acceleration=0. if config.platform_motion=='constant' else config.platform_accel_max
        # Conservative geometric budget, NOT a calibrated probability interval.
        return float(np.linalg.norm(position[:2]-state[:2])+position_std+
                     velocity_std*horizon+.5*acceleration*horizon**2),remaining

    def latch(self,center):
        self.terminal_position=np.asarray(center['position'],dtype=float).copy()
        self.terminal_velocity=np.asarray(center['velocity'],dtype=float).copy()
        self.terminal_velocity[2]=0.
        self.terminal_position_std=float(max(center.get('position_std_m',[0.,0.])[:2]))
        self.terminal_velocity_std=float(max(center.get('velocity_std_m_s',[0.,0.])[:2]))
        self.terminal_quality=float(center['confidence']); self.terminal_age=float(center['age_seconds'])
        self.terminal_blind_seconds=0.

    def terminal_estimate(self):
        if not self.terminal_active:
            return None
        return dict(mode='predicted',position=self.terminal_position.tolist(),
            velocity=self.terminal_velocity.tolist(),confidence=float(self.terminal_quality*np.exp(-self.terminal_blind_seconds/2.)),
            age_seconds=self.terminal_age+self.terminal_blind_seconds,near_field=True,
            estimate_kind='terminal_coast',terminal_seconds=self.terminal_seconds,
            reason='bounded_terminal_completion',source=f'Latched terminal estimate + horizontal velocity; max {self.terminal_duration:g}s finish controller, not a fresh detection')

    def apply(self,requested,state,center,dt,config):
        requested=np.asarray(requested,dtype=float)
        reliable=center.get('position') is not None and center.get('estimate_kind')!='terminal_coast'
        tracker_reliable=reliable
        if reliable:
            height=float(state[2]-center['position'][2])
            self.last_height=height
            self.last_plane=float(center['position'][2])
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
            return requested,dict(active=False,blend=0.,unknown_seconds=0.,descent_allowed=True,
                terminal_active=False,phase='network',terminal_timed_out=False)
        self.unknown_seconds=0. if reliable else self.unknown_seconds+dt
        height=self.last_height if self.last_height is not None else config.near_enter_height
        if self.last_plane is not None:
            height=float(state[2]-self.last_plane)
        terminal_timed_out=False; terminal_aborted=False
        completion_radius=None
        tilt=max(abs(v) for v in p.getEulerFromQuaternion(state[3:7])[:2])
        if config.terminal_finish_enabled:
            if self.terminal_active:
                self.terminal_seconds+=dt
                self.terminal_blind_seconds+=dt
                self.terminal_position+=dt*self.terminal_velocity
                # Fresh partial/full camera updates correct the coast; a mere KF
                # extrapolation must not reset its blind-time uncertainty budget.
                if (reliable and center.get('measurement_age_seconds',center.get('age_seconds',999.))<=.10
                        and center.get('confidence',0.)>=.12
                        and np.linalg.norm(np.asarray(center['position'])[:2]-self.terminal_position[:2])<.20):
                    self.latch(center); self.terminal_refreshes+=1
                error=np.linalg.norm(self.terminal_position[:2]-state[:2])
                relative_speed=np.linalg.norm(np.asarray(state[10:12])-self.terminal_velocity[:2])
                completion_radius,_=self.completion_radius(state,config)
                if self.terminal_seconds>=config.terminal_finish_seconds:
                    self.terminal_active=False; terminal_timed_out=True
                elif completion_radius>.40 or error>.30 or relative_speed>.35 or tilt>np.deg2rad(20) or state[2]-self.terminal_position[2]>.35 or requested[2]>.15:
                    self.terminal_active=False; terminal_aborted=True
            elif self.active and reliable and not self.terminal_attempted:
                distance=np.linalg.norm(np.asarray(center['position'])[:2]-state[:2])
                speed=np.linalg.norm(np.asarray(state[10:12])-np.asarray(center['velocity'])[:2])
                std=max(center.get('position_std_m',[float('inf')]*3)[:2])
                velocity_std=max(center.get('velocity_std_m_s',[0.,0.])[:2])
                completion_radius,remaining=self.completion_radius(state,config,center['position'],std,velocity_std,0.)
                if (.02<height<=config.terminal_entry_height and distance<=.16 and speed<=.12
                        and tilt<=np.deg2rad(12) and center.get('confidence',0.)>=.12 and std<=.30
                        and completion_radius<=.40 and remaining<config.terminal_finish_seconds
                        and requested[2]<-.02):
                    self.terminal_active=True; self.terminal_attempted=True; self.terminal_seconds=dt
                    self.terminal_duration=config.terminal_finish_seconds
                    self.latch(center)
            if self.terminal_active:
                center=self.terminal_estimate()
                reliable=True
                height=float(state[2]-center['position'][2])
        x=float(np.clip((height-.10)/.70,0.,1.))
        correction_limit=.08+.22*x*x*(3.-2.*x)
        platform_velocity=self.terminal_velocity[:2] if self.terminal_active else self.last_velocity[:2]*np.exp(-self.unknown_seconds/1.5)
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
        descent_limit=(.12+.12*x) if config.terminal_finish_enabled else (.08+.04*x)
        if self.terminal_active:
            descent_allowed=True
            target[2]=-config.terminal_finish_speed
        else:
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
            descent_allowed=descent_allowed,relative_speed_m_s=relative_speed,
            tracker_reliable=tracker_reliable,terminal_active=self.terminal_active,
            terminal_seconds=self.terminal_seconds,terminal_timed_out=terminal_timed_out,
            terminal_aborted=terminal_aborted,descent_limit_m_s=config.terminal_finish_speed if self.terminal_active else descent_limit,
            terminal_completion_radius_m=completion_radius,terminal_refreshes=self.terminal_refreshes,
            phase='terminal_finish' if self.terminal_active else 'near_tracking')


def near_center_reward(state,center):
    if not center.get('near_field') or center.get('position') is None:
        return 0.
    distance=np.linalg.norm(np.asarray(center['position'])[:2]-state[:2])
    relative_velocity=np.asarray(state[10:12])-np.asarray(center['velocity'])[:2]
    return float(center['confidence']*(.025*np.exp(-(distance/.12)**2)-.12*np.sum(relative_velocity**2)))
