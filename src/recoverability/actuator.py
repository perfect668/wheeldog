"""Sampled first-order response, continuous linear force ramps, command FIFO."""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class Actuator:
    force_max_N: float = 150.0
    command_max_N: float = 150.0
    rate_N_s: float = 1500.0
    response_tau_s: float = 0.04
    delay_s: float = 0.0

    def __post_init__(self):
        for k, v in asdict(self).items():
            if not math.isfinite(v) or v < 0 or (k in ('force_max_N', 'command_max_N', 'rate_N_s') and v == 0):
                raise ValueError(f"invalid actuator parameter {k}")
        if self.command_max_N != self.force_max_N:
            raise ValueError("EXP0 requires equal command and actual force bounds")

    def delay_steps(self, dt):
        n = int(round(self.delay_s/dt))
        if not math.isclose(n*dt, self.delay_s, abs_tol=1e-10):
            raise ValueError("delay_s must be an integer multiple of dt_s")
        return n

    def alpha(self, dt):
        return 1.0 if self.response_tau_s == 0 else -math.expm1(-dt/self.response_tau_s)


class ForceActuator:
    """Causal simulator. Governor bounds a command at its future arrival state.

    For delayed command q: f[k+1]=(1-alpha)f[k]+alpha*q.
    Force is linearly interpolated on [k*dt,(k+1)*dt].
    Queue initially zero; governor uses only its own known queued commands.
    """
    def __init__(self, parameters, dt):
        self.parameters, self.dt = parameters, dt
        self.alpha = parameters.alpha(dt)
        self.queue = [0.0] * parameters.delay_steps(dt)
        self.force = 0.0

    def advance(self, desired):
        if not math.isfinite(desired):
            raise ValueError("nonfinite command")
        p, a = self.parameters, self.alpha
        arrival_force = self.force
        for queued in self.queue:
            arrival_force = (1-a)*arrival_force + a*queued
        limit = p.rate_N_s*self.dt/a
        command = float(np.clip(desired, max(-p.command_max_N, arrival_force-limit),
                                min(p.command_max_N, arrival_force+limit)))
        self.queue.append(command)
        applied = self.queue.pop(0)
        start = self.force
        self.force = (1-a)*start+a*applied
        return start, self.force, command
