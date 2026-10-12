"""Validated height curriculum; height is measured above the platform deck."""
from dataclasses import replace

HEIGHT_LEVELS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def height_config(config, level):
    if not 0 <= level < len(HEIGHT_LEVELS):
        raise ValueError('Height level out of range')
    # Keep horizontal offsets, platform motion and control settings unchanged.
    # A common 60-second horizon allows the higher starts time to approach.
    return replace(config, initial_height=HEIGHT_LEVELS[level], offset_z=.3,
                   episode_seconds=60.)


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
