"""Independent posterior validation; never trust a solver status alone."""
import numpy as np

STATE_TOL = 2e-7  # m or m/s, applied to each explicitly named quantity
FORCE_TOL = 2e-5  # N (force-increment check is in N, not N/s)


def validate(model, actuator, disturbance, state, force, command, initial=(0.0, 0.0)):
    n = len(command)
    if state.shape != (n+1, 2) or force.shape != (n+1,):
        raise ValueError('invalid trajectory shape')
    if not all(np.all(np.isfinite(a)) for a in (state, force, command)):
        return {'valid_path_dynamics': False, 'terminal_met': False, 'nonfinite': True}
    jumps, d = disturbance.arrays(model, n)
    A, B0, B1, Bd = model.matrices()
    predicted = state[:-1] @ A.T + force[:-1,None]*B0 + force[1:,None]*B1 + d[:,None]*Bd
    predicted[:,1] += jumps[1:]
    dynamics = np.max(np.abs(state[1:]-predicted), axis=0)
    delay, alpha = actuator.delay_steps(model.dt_s), actuator.alpha(model.dt_s)
    delivered = np.zeros(n)
    if delay < n:
        delivered[delay:] = command[:n-delay]
    actuator_error = np.max(np.abs(force[1:] - (1-alpha)*force[:-1]-alpha*delivered))
    initial_error = np.abs(state[0] - (np.asarray(initial)+[0,jumps[0]]))
    extrema = np.array([model.interval_extrema(state[k], force[k], force[k+1], d[k]) for k in range(n)])
    peaks = np.maximum(np.max(np.abs(state), axis=0), np.max(extrema, axis=0))
    outside = np.flatnonzero((extrema[:,0] > model.x_safe_m+STATE_TOL)
                             | (extrema[:,1] > model.v_abs_max_m_s+STATE_TOL))
    outside_nodes = np.flatnonzero((np.abs(state[:,0]) > model.x_safe_m+STATE_TOL)
                                   | (np.abs(state[:,1]) > model.v_abs_max_m_s+STATE_TOL))
    first_outside = min([float(k*model.dt_s) for k in outside[:1]]
                        + [float(k*model.dt_s) for k in outside_nodes[:1]],default=None)
    result = {
        'contains_extrapolation_outside_path_domain': first_outside is not None,
        'first_violating_interval_start_s': first_outside,
        'nonfinite': False,
        'dynamics_x_error_m': float(dynamics[0]),
        'dynamics_v_error_m_s': float(dynamics[1]),
        'initial_x_error_m': float(initial_error[0]),
        'initial_v_error_m_s': float(initial_error[1]),
        'initial_force_error_N': float(abs(force[0])),
        'actuator_error_N': float(actuator_error),
        'force_violation_N': float(max(0,np.max(np.abs(force))-actuator.force_max_N)),
        'command_violation_N': float(max(0,np.max(np.abs(command))-actuator.command_max_N)),
        'increment_violation_N': float(max(0,np.max(np.abs(np.diff(force)))-actuator.rate_N_s*model.dt_s)),
        'peak_abs_x_m': float(peaks[0]), 'peak_abs_v_m_s': float(peaks[1]),
        'x_path_violation_m': float(max(0,peaks[0]-model.x_safe_m)),
        'v_path_violation_m_s': float(max(0,peaks[1]-model.v_abs_max_m_s)),
        'terminal_x_m': float(state[-1,0]), 'terminal_v_m_s': float(state[-1,1]),
        'terminal_force_N': float(force[-1]),
        'force_effort_N_s': float(np.sum(np.where(force[:-1]*force[1:] >= 0,
            (np.abs(force[:-1])+np.abs(force[1:]))/2,
            (force[:-1]**2+force[1:]**2)/(2*np.maximum(np.abs(force[:-1])+np.abs(force[1:]),1e-30))))*model.dt_s),
    }
    state_keys = ['dynamics_x_error_m','dynamics_v_error_m_s','initial_x_error_m',
                  'initial_v_error_m_s','x_path_violation_m','v_path_violation_m_s']
    force_keys = ['initial_force_error_N','actuator_error_N','force_violation_N',
                  'command_violation_N','increment_violation_N']
    result['valid_path_dynamics'] = bool(all(result[k] <= STATE_TOL for k in state_keys)
                                          and all(result[k] <= FORCE_TOL for k in force_keys))
    result['terminal_met'] = bool(abs(state[-1,0]) <= model.x_terminal_m+STATE_TOL
                                  and abs(state[-1,1]) <= model.v_terminal_m_s+STATE_TOL)
    return result
