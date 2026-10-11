"""Explicit curriculum for motion complexity; validation never uses easier mix."""
from dataclasses import replace


PHASE_NAMES=('constant straight','variable-speed straight','gentle random B-spline','faster random B-spline')


def motion_config(base_factory,phase,*assist_flags):
    if not 0<=phase<len(PHASE_NAMES):
        raise ValueError('Motion phase must be 0..3')
    cfg=base_factory(2 if phase==3 else 1,*assist_flags)
    settings=(dict(platform_motion='constant'),
              dict(platform_motion='variable_speed',platform_speed_min=.08,platform_speed_max=.35,
                   platform_accel_max=.10,platform_speed_noise=.25,
                   terminal_entry_height=.24,terminal_finish_seconds=2.4,terminal_finish_speed=.14),
              dict(platform_motion='random_bspline',platform_speed_min=.10,platform_speed_max=.40,
                   platform_accel_max=.15,platform_turn_rate_degrees=10.,platform_speed_noise=.30,
                   terminal_entry_height=.20,terminal_finish_seconds=2.0,terminal_finish_speed=.14),
              dict(platform_motion='random_bspline',platform_speed_min=.10,platform_speed_max=.50,
                   platform_accel_max=.25,platform_turn_rate_degrees=20.,platform_speed_noise=.40,
                   terminal_entry_height=.17,terminal_finish_seconds=1.6,terminal_finish_speed=.14))
    return replace(cfg,near_partial_tracking=cfg.near_landing_assist,near_controller_version=2,**settings[phase])


def promotion_ready(validations,phase,required_episodes):
    recent=[v for v in validations if v.get('motion_phase',0)==phase][-3:]
    if len(recent)<3 or required_episodes<20:
        return False
    if any(v['episodes']!=required_episodes or v['successes']/v['episodes']<.8
           or v['soft_landings']/v['episodes']<.6 for v in recent):
        return False
    rates=[v['successes']/v['episodes'] for v in recent]
    return max(rates)-min(rates)<=.0500001
