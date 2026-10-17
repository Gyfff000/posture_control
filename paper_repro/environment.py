"""Isolated Ladosz task: raw pixels and own state, no learned-policy assistance."""
import numpy as np
from landing_repro.environment import MovingPlatformLanding

class PaperLanding(MovingPlatformLanding):
    def __init__(self,cfg,gui=False,constant_descent=False):
        forbidden=[cfg.prediction_guidance,cfg.near_landing_assist,cfg.terminal_finish_enabled,
                   cfg.smooth_observation,cfg.action_smooth_weight,cfg.soft_landing_weight,
                   cfg.jerk_limit_xy,cfg.jerk_limit_z]
        if any(forbidden):raise ValueError('Assisted configuration is forbidden in paper task')
        if cfg.reward_mode!='paper_interpretation':raise ValueError('Paper reward required')
        self.constant_descent=constant_descent
        super().__init__(cfg,gui)
    def _observe(self):
        super()._observe()
        # Implementation choice: 3 x [scaled world velocity, xyzw quaternion].
        # Fig.3 establishes 21 additional inputs; the paper omits their exact encoding.
        self.proprio_frames[-1]=((np.clip(self.proprio_frames[-1],-1.,1.)+1.)*.5).astype(np.float32)
    def step(self,action):
        action=np.asarray(action,np.float32).copy()
        if self.constant_descent:action[2]=-1.
        return super().step(action)
