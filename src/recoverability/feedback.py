"""A fixed PD feedback comparator, with no access to future disturbance inputs."""
import time
import numpy as np
from .actuator import ForceActuator
from .metrics import validate
from .solver import Result


def simulate_feedback(model, actuator, disturbance, horizon_s, *, natural_frequency_s_inv=4.0, initial=(0.0,0.0)):
    start = time.perf_counter()
    n = model.steps(horizon_s)
    jumps,d = disturbance.arrays(model,n)  # environment only; never supplied to feedback rule
    state, force, command = np.zeros((n+1,2)),np.zeros(n+1),np.zeros(n)
    state[0] = np.asarray(initial)+[0,jumps[0]]
    act = ForceActuator(actuator,model.dt_s)
    A,B0,B1,Bd = model.matrices()
    wn = natural_frequency_s_inv
    try:
        with np.errstate(over='raise',invalid='raise'):
            for k in range(n):
                # Perfect current x,v sensing. All configured latency is in command FIFO.
                desired = -model.mass_kg*((model.omega**2+wn**2)*state[k,0]+2*wn*state[k,1])
                f0,f1,command[k] = act.advance(desired)
                force[k+1] = f1
                state[k+1] = A@state[k]+B0*f0+B1*f1+Bd*d[k]
                state[k+1,1] += jumps[k+1]
            metrics = validate(model,actuator,disturbance,state,force,command,initial)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        return Result('SIMULATION_ERROR',f'{type(exc).__name__}: {exc}',time.perf_counter()-start)
    numerical_ok = not metrics.get('nonfinite',False)
    # Path violations are an observed failure of this policy, not proof of infeasibility.
    recovered = metrics['valid_path_dynamics'] and metrics['terminal_met']
    status = ('RECOVERED' if recovered else 'NOT_RECOVERED') if numerical_ok else 'SIMULATION_ERROR'
    return Result(status,'causal fixed PD; NOT_RECOVERED is not an infeasibility certificate',
                  time.perf_counter()-start,state,force,command,metrics)
