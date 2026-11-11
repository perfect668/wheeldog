"""SI-unit LIPM, exact integration for piecewise-linear actual force."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class Model:
    mass_kg: float = 70.0
    height_m: float = 1.0
    gravity_m_s2: float = 9.81
    dt_s: float = 0.02
    x_safe_m: float = 0.20
    v_abs_max_m_s: float = 2.0
    x_terminal_m: float = 0.01
    v_terminal_m_s: float = 0.03

    def __post_init__(self):
        for k, v in asdict(self).items():
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{k} must be finite and positive")
        if self.x_terminal_m > self.x_safe_m or self.v_terminal_m_s > self.v_abs_max_m_s:
            raise ValueError("terminal box must lie within path box")

    @property
    def omega(self):
        return math.sqrt(self.gravity_m_s2 / self.height_m)

    def steps(self, seconds):
        n = int(round(seconds / self.dt_s))
        if n < 1 or not math.isclose(n * self.dt_s, seconds, abs_tol=1e-10):
            raise ValueError("time must be a positive integer multiple of dt_s")
        return n

    def matrices(self, duration=None):
        """s_next=A s+B0 f_start+B1 f_end+Bd d; d constant per interval."""
        t = self.dt_s if duration is None else duration
        # Augmented state [x, v, f, force_slope, d]; exact continuous flow.
        M = np.zeros((5, 5))
        M[0, 1] = 1
        M[1, 0] = self.omega**2
        M[1, 2] = M[1, 4] = 1 / self.mass_kg
        M[2, 3] = 1
        E = expm(M * t)
        return E[:2, :2], E[:2, 2] - E[:2, 3] / t, E[:2, 3] / t, E[:2, 4]

    def step(self, state, f_start, f_end, disturbance=0.0):
        A, B0, B1, Bd = self.matrices()
        return A @ state + B0 * f_start + B1 * f_end + Bd * disturbance

    def at_time(self, state, f_start, f_end, disturbance, t):
        """Exact state at t within one interval, used for independent path checks."""
        if not 0 <= t <= self.dt_s:
            raise ValueError("time outside interval")
        w = self.omega
        r = (f_end - f_start) / self.dt_s
        c, s = math.cosh(w * t), math.sinh(w * t)
        q = (f_start + disturbance) / self.mass_kg
        x, v = state
        return np.array([c*x+s*v/w+(c-1)*q/w**2+(s/w-t)*r/(self.mass_kg*w**2),
                         w*s*x+c*v+s*q/w+(c-1)*r/(self.mass_kg*w**2)])

    def interval_extrema(self, state, f_start, f_end, disturbance):
        """Exact x and v extrema: velocity roots reduce to a quadratic in exp(w t)."""
        w = self.omega
        r = (f_end-f_start) / self.dt_s
        a = state[1] + r / (self.mass_kg*w**2)
        b = w*state[0] + (f_start+disturbance) / (self.mass_kg*w)
        c = -r / (self.mass_kg*w**2)
        times = [0.0, self.dt_s]
        coeff = np.trim_zeros(np.array([(a+b)/2, c, (a-b)/2]), 'f')
        if len(coeff) > 1:
            for z in np.roots(coeff):
                if abs(z.imag) < 1e-9 and 1 < z.real < math.exp(w*self.dt_s):
                    times.append(math.log(z.real)/w)
        # Acceleration roots locate velocity extrema.
        if abs(a+b) > 1e-14 and (a-b)/(a+b) > 0:
            z = math.sqrt((a-b)/(a+b))
            if 1 < z < math.exp(w*self.dt_s):
                times.append(math.log(z)/w)
        values = np.array([self.at_time(state, f_start, f_end, disturbance, t) for t in times])
        return np.max(np.abs(values), axis=0)
