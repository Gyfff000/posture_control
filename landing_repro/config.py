from dataclasses import asdict, dataclass


@dataclass
class EnvConfig:
    physics_hz: int = 300
    control_hz: int = 60
    policy_hz: int = 30
    episode_seconds: float = 30.0
    initial_height: float = 10.0
    offset_xy: float = 3.0
    offset_z: float = 3.0
    platform_speed_min: float = 0.0
    platform_speed_max: float = 1.0
    platform_yaw_degrees: float = 0.0
    platform_top: float = 0.3
    camera_fov: float = 90.0
    camera_offset_x: float = 0.045
    camera_offset_z: float = -0.008
    camera_near: float = 0.001
    visible_pixels: int = 1
    horizontal_reward: str = "normalized_interpretation"
    terminal_overrides_visibility: bool = True
    visibility_gate: bool = True
    guard_actual_vertical_velocity: bool = True
    acceleration_limit_xy: float = 3.0
    acceleration_limit_z: float = 1.0
    reward_mode: str = "paper_interpretation"
    escape_distance: float = 0.0
    lost_view_seconds: float = 0.0
    teacher_descent_fast: float = .35
    smooth_observation: bool = False
    action_smooth_weight: float = 0.0
    soft_landing_weight: float = 0.0
    jerk_limit_xy: float = 0.0
    jerk_limit_z: float = 0.0
    velocity_response_seconds: float = .10

    def to_dict(self):
        return asdict(self)


@dataclass
class AgentConfig:
    proprio_dim: int = 21
    seed: int = 1
    learning_rate: float = 0.0001
    tau: float = 0.0012
    discount: float = 0.99
    feature_dim: int = 50
    hidden_dim: int = 512
    batch_size: int = 16
    replay_capacity: int = 5000
    warmup_steps: int = 256
    update_every: int = 4
    n_step: int = 3
    noise_start: float = 1.0
    noise_end: float = 0.1
    noise_decay_steps: int = 50000
    noise_clip: float = 0.3
    augmentation_padding: str = "replicate"
    torch_threads: int = 2

    def to_dict(self):
        return asdict(self)
