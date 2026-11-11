"""Explicit velocity jumps (m/s) and separate piecewise constant forces (N)."""
from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class Disturbance:
    kind: str = 'velocity_impulse'
    onset_s: float = 0.0
    delta_v_m_s: float = 0.0
    force_N: float = 0.0
    duration_s: float = 0.0

    def arrays(self, model, n):
        values = asdict(self)
        if any(not math.isfinite(v) for k,v in values.items() if k != 'kind'):
            raise ValueError("nonfinite disturbance")
        j = int(round(self.onset_s/model.dt_s))
        if self.onset_s < 0 or not math.isclose(j*model.dt_s, self.onset_s, abs_tol=1e-10) or j >= n:
            raise ValueError("onset must be a grid node before horizon")
        jumps, force = np.zeros(n+1), np.zeros(n)
        if self.kind == 'velocity_impulse':
            if self.force_N != 0 or self.duration_s != 0:
                raise ValueError("velocity impulse must not include force or duration")
            jumps[j] = self.delta_v_m_s
        elif self.kind == 'force_pulse':
            if self.delta_v_m_s != 0:
                raise ValueError("force pulse must not include delta_v")
            length = model.steps(self.duration_s)
            if j+length > n:
                raise ValueError("pulse ends after horizon")
            force[j:j+length] = self.force_N
        else:
            raise ValueError("unknown disturbance kind")
        return jumps, force
