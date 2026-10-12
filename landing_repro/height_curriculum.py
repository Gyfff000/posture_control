"""Validated height curriculum; height is measured above the platform deck."""
from dataclasses import replace
import math

HEIGHT_LEVELS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)


def height_config(config, level):
    if not 0 <= level < len(HEIGHT_LEVELS):
        raise ValueError('Height level out of range')
    # Keep horizontal offsets, platform motion and control settings unchanged.
    # Allow descent at 0.25 m/s plus 20 s for alignment; preserve low stages.
    height=HEIGHT_LEVELS[level]
    horizon=max(60.,float(math.ceil((height+.3)/.25+20.)))
    # The actor still receives 84x84; this cap applies to auxiliary RGB only.
    auxiliary_cap=84 if height<3. else int(math.ceil((height+.3)*56/84)*84)
    return replace(config, initial_height=height, offset_z=.3,
                   episode_seconds=horizon, center_camera_size=auxiliary_cap)


def height_promotion_ready(validations, phase, level, required_episodes=20):
    if required_episodes < 20:
        return False
    recent = [v for v in validations
              if v.get('motion_phase') == phase and v.get('height_level') == level][-3:]
    if len(recent) < 3:
        return False
    if any(v['episodes'] != required_episodes or
           v['successes'] / v['episodes'] < .8 or
           v['soft_landings'] / v['episodes'] < .6 for v in recent):
        return False
    rates = [v['successes'] / v['episodes'] for v in recent]
    return max(rates) - min(rates) <= .0500001
