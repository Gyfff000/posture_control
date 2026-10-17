"""Published parameters, distinct from CPU feasibility and author ambiguities."""
from dataclasses import replace
from landing_repro.config import AgentConfig, EnvConfig

LADOSZ_PAPER = {
 'title':'Autonomous Landing on a Moving Platform Using Vision-Based Deep Reinforcement Learning',
 'doi':'10.1109/LRA.2024.3379837','year':2024,
 'table_1':{'episodes':4000,'episode_seconds':30,'max_episode_steps':900,
 'training_seeds':[1,2,3,4,5],'aggregated_steps':10,'safe_descent_m_s':-.5,
 'image_shape':[3,84,84],'learning_rate':.005,'tau':.0012,'batch_size':1024,
 'replay_capacity':300000,'discount':.99,'warmup_steps':10000,
 'automatic_entropy_tuning':'yes (unresolved: standard DrQ-v2 has no entropy temperature)',
 'reward':{'Rp':-.01,'alpha':.4,'beta':1.,'gamma':1.,'zeta':30.,'rho':20.,'ds_m':10.,'Rs':120.}},
 'figure_3':{'encoder_output':39200,'own_state_stack':21,'feature_dim':50,'hidden_dim':512,'normalization':'BatchNorm1d'},
 'evaluation':{'during_training_every_episodes':100,'during_training_episodes':10,'final_episodes':500},
 'unresolved':['Eq9 literal exponent ds-dxy/ds versus normalized (ds-dxy)/ds',
 'Eq8 bounds the commanded velocity but exponentiates actual velocity; severe overspeed gives unbounded reward',
 'Fig3 BatchNorm1d versus official DrQ-v2 LayerNorm; both must be distinguished',
 'automatic entropy tuning listed for DrQ-v2 without defining temperature objective',
 'aggregated steps and physical/control clocks not fully specified',
 'which quaternion convention and normalization of own state were used',
 'camera calibration and exact base controller gains not provided',
 'whether terminal contact rewards override the LOS outer gate',
 'original pad photograph and selected 400 DTD textures unavailable locally']}

PERCHRL_PAPER = {
 'title':'PerchRL: Vision-Based Agile Perching on Inclined Platforms under Rapid and Irregular Motion',
 'arxiv':'2606.03441v3','source':'https://arxiv.org/html/2606.03441v3',
 'algorithm':'PPO','simulator':'OmniDrones / Isaac Sim','parallel_envs':8192,
 'physics_hz':100,'episode_steps':500,'future_critic_observations':15,
 'stages':['state-based pretraining','visibility-gated pose-observation fine-tuning'],
 'initialization':{'platform_speed_m_s':[0.,5.],'inclination_degrees':[0.,90.],
 'relative_altitude_m':[1.2,2.],'horizontal_distance_m':[1.2,3.5],
 'azimuth_offset_degrees':[-60.,60.],'roll_pitch_rad':[-.6,.6],
 'horizontal_velocity_m_s':[-.1,.1],'vertical_velocity_m_s':[-.2,.2],
 'roll_pitch_rate_rad_s':[-.2,.2],'yaw_rate_rad_s':[-.1,.1]},
 'randomization':{'mass_fraction':.30,'inertia_fraction':.10,'thrust_to_weight_fraction':.15,
 'camera_tilt_degrees':[-5.,5.],'detector_hz':[20.,60.]},
 'experiments':{'state':['PerchRL','w/o temporal augmentation','Fast-Perching','InclineLander'],
 'trajectories':['reciprocating linear','racetrack','figure-8','random B-spline'],
 'vision':['full model','LSTM baseline','input augmentation / active perception ablations'],
 'baseline_training_steps_up_to':40000000},
 'unresolved':['official PerchRL task repository not linked by checked paper/author page as of 2026-10-17',
 'PPO hyperparameters, actor TCN channels/kernels/history and reward weights incomplete',
 'nominal robot parameters, camera calibration and contact tolerances not fully specified',
 'exact pose noise model and trajectory geometry settings require upstream code or clarification']}


def ladosz_configs(seed=1, reward_interpretation='normalized', no_los=False, vertical_reward='actual_guard'):
    env=replace(EnvConfig(), initial_height=10.,offset_xy=3.,offset_z=3.,
        episode_seconds=30.,platform_motion='constant',platform_speed_min=0.,platform_speed_max=1.,
        platform_yaw_degrees=0.,camera_fov=90.,center_camera_size=84,
        horizontal_reward='normalized_interpretation' if reward_interpretation=='normalized' else 'literal',
        terminal_overrides_visibility=False,guard_actual_vertical_velocity=vertical_reward=='actual_guard',
        visibility_gate=not no_los,acceleration_limit_xy=1e9,acceleration_limit_z=1e9,
        prediction_guidance=False,near_landing_assist=False,terminal_finish_enabled=False,
        smooth_observation=False,action_smooth_weight=0.,soft_landing_weight=0.)
    agent=replace(AgentConfig(),seed=seed,learning_rate=.005,tau=.0012,
        batch_size=1024,replay_capacity=300000,warmup_steps=10000,update_every=2,
        feature_dim=50,hidden_dim=512,augmentation_padding='constant',torch_threads=2)
    return env,agent
