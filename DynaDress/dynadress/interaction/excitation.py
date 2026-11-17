"""Bounded deterministic exploration layered over the preserved M1.1 servo.

The schedule starts on entry to insert. It never reads K, D or friction.
Offsets are absolute, smooth pulses (not accumulated position drift).
"""
import numpy as np


def command(profile, elapsed):
    output = dict(primitive='advance', speed_scale=1., differential_m=0., roll_rad=0.)
    if profile == 'baseline' or elapsed is None:
        return output
    if profile not in ('probe_a', 'probe_b', 'probe_c'):
        raise ValueError(f'Unknown exploration profile: {profile}')
    shift = {'probe_a': 0., 'probe_b': .65, 'probe_c': 1.3}[profile]
    t = elapsed - shift
    if 1.0 <= t < 1.65:
        output.update(primitive='pause', speed_scale=0.)
    elif 3.0 <= t < 3.5:
        output.update(primitive='retreat', speed_scale=-.45)
    elif 5.0 <= t < 7.0:
        pulse = np.sin(np.pi * (t - 5.) / 2.) ** 2
        sign = -1. if profile == 'probe_b' else 1.
        output.update(primitive='roll_differential', speed_scale=.65,
                      differential_m=sign*.002*pulse,
                      roll_rad=sign*np.radians(2.)*pulse)
    return output
