"""Real Crazyflie physics, kinematic collision platform and body-fixed camera.

No platform truth appears in policy observations. See README for assumptions.
"""
from collections import deque
from pathlib import Path
import tempfile

import cv2
import gymnasium as gym
import numpy as np
import pybullet as p
from PIL import Image, ImageDraw
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from .config import EnvConfig


LIMITS = np.array([1.5, 1.5, 0.5], dtype=np.float32)


def reward_terms(distance, actual_vz, commanded_vz, visible, event, cfg):
    """Equations 7-10, with an explicit switch for the ambiguous equation 9."""
    rp, zeta, rho, ds = -0.01, 30.0, 20.0, 10.0
    if (commanded_vz < -0.5 - 1e-7 or commanded_vz > 1e-7
            or (cfg.guard_actual_vertical_velocity and actual_vz < -0.5 - 1e-7)):
        rz = rp
    else:
        # Guard the exponent only against floating point overflow; report excursions.
        exponent = float(np.clip(actual_vz / -0.5, -50, 50))
        rz = (zeta ** exponent - 1) / (zeta - 1)
    if distance > ds:
        rxy = 0.0
    else:
        exponent = ((ds - distance) / ds if cfg.horizontal_reward == "normalized_interpretation"
                    else ds - distance / ds)
        rxy = (rho ** exponent - 1) / (rho - 1)
    rl = 120.0 if event == "success" else rp if event in ("ground", "platform_edge", "invalid") else 0.0
    total = rz + 0.4 * rxy + rl if visible or not cfg.visibility_gate else rp
    if cfg.terminal_overrides_visibility and event != "running":
        total = rl
    return {"r_z": float(rz), "r_xy": float(rxy), "r_landing": rl, "reward": float(total)}


def make_assets():
    root = Path(tempfile.gettempdir()) / "ladosz_landing_assets_v1"
    root.mkdir(exist_ok=True)
    texture = root / "helipad.png"
    if not texture.exists():
        im = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(im)
        draw.rectangle([4, 4, 507, 507], outline="black", width=12)
        draw.ellipse([60, 60, 452, 452], outline="black", width=22)
        draw.rectangle([166, 140, 202, 372], fill="black")
        draw.rectangle([310, 140, 346, 372], fill="black")
        draw.rectangle([166, 238, 346, 274], fill="black")
        im.save(texture)
    mesh = root / "platform_top.obj"
    if not mesh.exists():
        mesh.write_text("v -0.5 -0.5 0.04\nv 0.5 -0.5 0.04\nv 0.5 0.5 0.04\nv -0.5 0.5 0.04\n"
                        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nf 1/1 2/2 3/3\nf 1/1 3/3 4/4\n")
    return mesh, texture


class PlatformAviary(CtrlAviary):
    """Move a mass-zero kinematic platform at every individual physics tick."""
    platform_id = None

    def _physics(self, rpm, nth_drone):
        if self.platform_id is not None and nth_drone == 0:
            t = self.platform_tick * self.PYB_TIMESTEP
            pos = self.platform_origin + self.platform_velocity * t
            p.resetBasePositionAndOrientation(self.platform_id, pos, self.platform_quat,
                                              physicsClientId=self.CLIENT)
            p.resetBaseVelocity(self.platform_id, self.platform_velocity, [0, 0, 0],
                                physicsClientId=self.CLIENT)
            self.platform_tick += 1
        super()._physics(rpm, nth_drone)


class MovingPlatformLanding(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg=None, gui=False):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        c = self.cfg
        assert c.physics_hz % c.control_hz == 0 and c.control_hz % c.policy_hz == 0
        self.gui = gui
        self.sim = PlatformAviary(drone_model=DroneModel.CF2X, num_drones=1,
                                 initial_xyzs=np.array([[0., 0., c.initial_height + c.platform_top]]),
                                 physics=Physics.PYB, pyb_freq=c.physics_hz, ctrl_freq=c.control_hz,
                                 gui=gui, record=False, obstacles=False, user_debug_gui=False)
        self.client = self.sim.getPyBulletClient()
        self.drone = int(self.sim.DRONE_IDS[0])
        self.action_space = gym.spaces.Box(-1., 1., (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "pixels": gym.spaces.Box(0, 255, (3, 84, 84), dtype=np.uint8),
            "proprio": gym.spaces.Box(-np.inf, np.inf, (30 if c.smooth_observation else 21,), dtype=np.float32)})
        self.frames, self.proprio_frames = deque(maxlen=3), deque(maxlen=3)
        self.closed = False
        self.finished = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        c = self.cfg
        xyz = np.array([self.np_random.uniform(-c.offset_xy, c.offset_xy),
                        self.np_random.uniform(-c.offset_xy, c.offset_xy),
                        c.platform_top + c.initial_height + self.np_random.uniform(-c.offset_z, c.offset_z)])
        xyz = np.asarray(options.get("initial_xyz", xyz), dtype=float)
        speed = float(options.get("platform_speed", self.np_random.uniform(c.platform_speed_min, c.platform_speed_max)))
        yaw = float(options.get("platform_yaw", np.deg2rad(self.np_random.uniform(-c.platform_yaw_degrees, c.platform_yaw_degrees))))
        self.sim.platform_id = None
        self.sim.INIT_XYZS = xyz.reshape(1, 3)
        self.sim.reset(seed=seed)
        self.drone = int(self.sim.DRONE_IDS[0])
        self.controller = DSLPIDControl(DroneModel.CF2X)
        mesh, texture = make_assets()
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.5, .5, .04], physicsClientId=self.client)
        visual = p.createVisualShape(p.GEOM_MESH, fileName=str(mesh), rgbaColor=[1, 1, 1, 1],
                                     flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.client)
        self.platform_yaw = yaw
        self.platform_quat = p.getQuaternionFromEuler([0, 0, yaw])
        self.platform_origin = np.array([0., 0., c.platform_top - .04])
        self.platform_velocity = np.array([speed * np.cos(yaw), speed * np.sin(yaw), 0.])
        self.platform_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                                            baseVisualShapeIndex=visual, basePosition=self.platform_origin,
                                            baseOrientation=self.platform_quat, physicsClientId=self.client)
        tid = p.loadTexture(str(texture), physicsClientId=self.client)
        p.changeVisualShape(self.platform_id, -1, textureUniqueId=tid, physicsClientId=self.client)
        p.changeDynamics(self.platform_id, -1, lateralFriction=0.8, restitution=0,
                         physicsClientId=self.client)
        self.sim.platform_id = self.platform_id
        self.sim.platform_origin = self.platform_origin
        self.sim.platform_velocity = self.platform_velocity
        self.sim.platform_quat = self.platform_quat
        self.sim.platform_tick = 0
        self.steps, self.elapsed, self.finished = 0, 0.0, False
        self.invisible_seconds = 0.
        if hasattr(self,'_center_tracker'):
            self._center_tracker.reset()
        self.applied_velocity = np.zeros(3)
        self.previous_command = np.zeros(3, np.float32)
        self.command_acceleration = np.zeros(3)
        self.frames.clear()
        self.proprio_frames.clear()
        self._observe()
        while len(self.frames) < 3:
            self.frames.append(self.frames[0].copy())
            self.proprio_frames.append(self.proprio_frames[0].copy())
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            self.update_debug_camera()
        return self._observation(), self._info("running", np.zeros(3))

    @property
    def state(self):
        return self.sim._getDroneStateVector(0).copy()

    @property
    def platform_position(self):
        return np.array(p.getBasePositionAndOrientation(self.platform_id, physicsClientId=self.client)[0])

    def _observe(self):
        state = self.state
        rotation = np.array(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
        eye = state[:3] + rotation @ np.array([self.cfg.camera_offset_x, 0., self.cfg.camera_offset_z])
        direction, up = rotation @ np.array([0., 0., -1.]), rotation @ np.array([1., 0., 0.])
        self.camera_view = p.computeViewMatrix(eye, eye + direction, up)
        projection = p.computeProjectionMatrixFOV(self.cfg.camera_fov, 1., self.cfg.camera_near, 100.)
        _, _, rgba, _, seg = p.getCameraImage(84, 84, self.camera_view, projection,
                                             renderer=p.ER_TINY_RENDERER,
                                             flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                             physicsClientId=self.client)
        self.rgb = np.asarray(rgba, np.uint8).reshape(84, 84, 4)[..., :3]
        self.seg = np.asarray(seg, np.int32).reshape(84, 84)
        self.visible_count = int(np.count_nonzero((self.seg & ((1 << 24) - 1)) == self.platform_id))
        self.visible = self.visible_count >= self.cfg.visible_pixels
        self.frames.append(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY))
        quat = state[3:7].copy()
        if quat[3] < 0:
            quat *= -1
        own_state = np.concatenate([state[10:13] / LIMITS, quat]).astype(np.float32)
        self.proprio_frames.append(own_state)
        if self.cfg.prediction_guidance:
            from .center_marker import estimate_center
            self.center_estimate=estimate_center(self)

    def _observation(self):
        own = np.concatenate(self.proprio_frames)
        if self.cfg.smooth_observation:
            limits = np.array([self.cfg.acceleration_limit_xy]*2 + [self.cfg.acceleration_limit_z])
            own = np.r_[own, self.previous_command, self.applied_velocity/LIMITS,
                        self.command_acceleration/limits].astype(np.float32)
        return {"pixels": np.stack(self.frames), "proprio": own}

    def fully_inside(self):
        pos = self.platform_position
        co, si = np.cos(self.platform_yaw), np.sin(self.platform_yaw)
        rotation = np.array([[co, si], [-si, co]])
        for link in range(-1, p.getNumJoints(self.drone, physicsClientId=self.client)):
            lo, hi = p.getAABB(self.drone, link, physicsClientId=self.client)
            corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
            local = (rotation @ (corners - pos[:2]).T).T
            if np.any(np.abs(local) > .5 - 1e-4):
                return False
        return True

    def _event(self):
        contacts = p.getContactPoints(self.drone, self.platform_id, physicsClientId=self.client)
        if contacts:
            top = any(pt[7][2] > .5 and pt[6][2] >= self.cfg.platform_top - .03 for pt in contacts)
            return "success" if top and self.fully_inside() else "platform_edge"
        if p.getContactPoints(self.drone, self.sim.PLANE_ID, physicsClientId=self.client):
            return "ground"
        if not np.isfinite(self.state).all() or self.state[2] < -.1:
            return "invalid"
        return "running"

    def _info(self, event, command):
        state = self.state
        rel = state[:3] - (self.platform_position + np.array([0., 0., .04]))
        return {"event": event, "success": event == "success", "time_s": self.elapsed,
                "visible": self.visible, "visible_pixels": self.visible_count,
                "position": state[:3].tolist(), "platform_position": self.platform_position.tolist(),
                "platform_velocity": self.platform_velocity.tolist(), "relative_position": rel.tolist(),
                "relative_velocity": (state[10:13] - self.platform_velocity).tolist(),
                "rpy": state[7:10].tolist(), "command_velocity": command.tolist()}

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset() after an episode ends")
        action = np.asarray(action, np.float32)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Expected three finite normalized velocity commands")
        command = np.clip(action, -1, 1) * LIMITS
        action_change = float(np.mean((np.clip(action, -1, 1) - self.previous_command)**2))
        before_relative = self.state[:3] - (self.platform_position + np.array([0., 0., .04]))
        before_position=self.state[:3].copy()
        before_time=self.elapsed
        previous_estimate=getattr(self,'center_estimate',{})
        event = "running"
        pre_contact_velocity = None
        for _ in range(self.cfg.control_hz // self.cfg.policy_hz):
            state = self.state
            acceleration = np.array([self.cfg.acceleration_limit_xy, self.cfg.acceleration_limit_xy,
                                     self.cfg.acceleration_limit_z])
            if self.cfg.jerk_limit_xy > 0 and self.cfg.jerk_limit_z > 0:
                dt = self.sim.CTRL_TIMESTEP
                desired = np.clip((command-self.applied_velocity)/self.cfg.velocity_response_seconds,
                                  -acceleration, acceleration)
                jerk = np.array([self.cfg.jerk_limit_xy]*2 + [self.cfg.jerk_limit_z])
                self.command_acceleration += np.clip(desired-self.command_acceleration, -jerk*dt, jerk*dt)
                self.applied_velocity += self.command_acceleration*dt
            else:
                delta = np.clip(command-self.applied_velocity,
                                -acceleration*self.sim.CTRL_TIMESTEP, acceleration*self.sim.CTRL_TIMESTEP)
                self.applied_velocity += delta
                self.command_acceleration = delta/self.sim.CTRL_TIMESTEP
            rpm, _, _ = self.controller.computeControl(
                control_timestep=self.sim.CTRL_TIMESTEP, cur_pos=state[:3], cur_quat=state[3:7],
                cur_vel=state[10:13], cur_ang_vel=state[13:16], target_pos=state[:3],
                target_rpy=np.zeros(3), target_vel=self.applied_velocity)
            self.sim.step(rpm.reshape(1, 4))
            self.elapsed += self.sim.CTRL_TIMESTEP
            event = self._event()
            if event != "running":
                pre_contact_velocity = state[10:13] - self.platform_velocity
                break
        self.steps += 1
        self._observe()
        info = self._info(event, command)
        self.invisible_seconds = 0. if self.visible else self.invisible_seconds + 1./self.cfg.policy_hz
        info["applied_velocity"] = self.applied_velocity.tolist()
        info["pre_contact_relative_velocity"] = (None if pre_contact_velocity is None else pre_contact_velocity.tolist())
        info['command_acceleration'] = self.command_acceleration.tolist()
        info['action_change_squared'] = action_change
        dxy = float(np.linalg.norm(info["relative_position"][:2]))
        info.update(reward_terms(dxy, self.state[12], command[2], self.visible, event, self.cfg))
        if self.cfg.reward_mode == "guided_progress":
            before_potential = np.linalg.norm(before_relative[:2]) + .4 * max(0., before_relative[2])
            potential = dxy + .4 * max(0., info["relative_position"][2])
            progress = 4. * (before_potential - potential)
            tracking = (.04 if dxy < .2 else -.02 * min(dxy, 5.))
            unsafe_descent = -.08 if dxy > .3 and command[2] < -.1 else 0.
            info["reward"] = float(progress + tracking + unsafe_descent - .005)
            if event == "success":
                info["reward"] = 10.
            elif event != "running":
                info["reward"] = -10.
            info.update(progress_reward=float(progress), tracking_reward=tracking,
                        unsafe_descent_penalty=unsafe_descent)
        info['prediction_reward']=0.
        if self.cfg.prediction_guidance:
            from .center_marker import prediction_progress
            center=self.center_estimate
            info['center_estimate']=center
            # Detector loss includes clipped/partial markers even when some platform pixels remain.
            # Replace privileged dense guidance in this branch; retain physical terminal outcomes.
            if center['mode']!='observed' and self.cfg.reward_mode=='guided_progress' and event=='running':
                predicted_reward=prediction_progress(before_position,self.state[:3],previous_estimate,
                    center,self.elapsed-before_time,self.cfg.prediction_reward_weight)
                info.update(prediction_reward=predicted_reward,progress_reward=0.,tracking_reward=0.,
                    unsafe_descent_penalty=0.,reward=predicted_reward-.005)
        if self.cfg.escape_distance > 0 and dxy > self.cfg.escape_distance and event == "running":
            event = info["event"] = "escaped"
            info["reward"] = -10.
        prediction_valid=self.cfg.prediction_guidance and self.center_estimate['mode']=='predicted'
        if self.cfg.lost_view_seconds > 0 and self.invisible_seconds >= self.cfg.lost_view_seconds and event == "running" and not prediction_valid:
            event = info["event"] = "lost_view"
            info["reward"] = -5.
        terminated = event != "running"
        truncated = not terminated and self.elapsed >= self.cfg.episode_seconds - 1e-8
        if truncated:
            info["event"] = "timeout"
            if self.cfg.reward_mode == "guided_progress":
                info["reward"] = -2.
        smooth_penalty = self.cfg.action_smooth_weight * action_change
        soft_penalty = 0.
        relative = np.asarray(info['relative_velocity'])
        if self.cfg.soft_landing_weight > 0 and 0 < info['relative_position'][2] < .6 and event == 'running':
            soft_penalty = self.cfg.soft_landing_weight * (
                np.sum(relative[:2]**2) + 2.*max(0., -relative[2]-.15)**2)
        contact = pre_contact_velocity
        info['soft_landing'] = bool(info['success'] and contact is not None
            and np.linalg.norm(contact[:2]) <= .2 and abs(contact[2]) <= .18
            and np.max(np.abs(self.state[7:9])) <= np.deg2rad(10))
        if info['success'] and self.cfg.soft_landing_weight > 0:
            # Preserve geometric success; separately reward the quality of first contact.
            soft_penalty += 2.*(np.linalg.norm(contact[:2])**2 + max(0., abs(contact[2])-.15)**2)
        info['smooth_penalty'], info['soft_penalty'] = smooth_penalty, soft_penalty
        info['reward'] -= smooth_penalty + soft_penalty
        self.previous_command = np.clip(action, -1, 1).copy()
        self.finished = terminated or truncated
        return self._observation(), info["reward"], terminated, truncated, info

    def update_debug_camera(self):
        state = self.state
        focus = getattr(self, "debug_camera_focus", "overview")
        if focus == "drone":
            p.resetDebugVisualizerCamera(.32, 45, -25, state[:3], physicsClientId=self.client)
            return
        if focus == "platform":
            p.resetDebugVisualizerCamera(1.65, 45, -55, self.platform_position, physicsClientId=self.client)
            return
        center = (state[:3] + self.platform_position) / 2
        distance = max(1.1, np.linalg.norm(state[:3] - self.platform_position) * 1.2)
        p.resetDebugVisualizerCamera(distance, 40, -27, center, physicsClientId=self.client)

    def render(self):
        return self.rgb.copy()

    def close(self):
        if not self.closed:
            if p.isConnected(self.client):
                self.sim.close()
            self.closed = True
