"""Sparse linear-programming existence oracle; offline known disturbance only."""
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, diags
from .metrics import validate


class Status(str, Enum):
    RECOVERABLE = 'RECOVERABLE'
    INFEASIBLE = 'INFEASIBLE'
    SOLVER_ERROR = 'SOLVER_ERROR'


@dataclass
class Result:
    status: str
    reason: str
    elapsed_s: float
    state: np.ndarray = field(default_factory=lambda: np.empty((0,2)))
    force: np.ndarray = field(default_factory=lambda: np.empty(0))
    command: np.ndarray = field(default_factory=lambda: np.empty(0))
    metrics: dict = field(default_factory=dict)
    solver_status: int | None = None
    solver_time_s: float = 0.0


class SparseRows:
    def __init__(self, width):
        self.width, self.row, self.col, self.val, self.rhs = width, [], [], [], []

    def add(self, entries, rhs):
        i = len(self.rhs)
        for j, v in entries.items():
            if v:
                self.row.append(i); self.col.append(j); self.val.append(v)
        self.rhs.append(rhs)

    def matrix(self):
        return coo_matrix((self.val,(self.row,self.col)),shape=(len(self.rhs),self.width)).tocsr()


def solve_oracle(model, actuator, disturbance, horizon_s, *, initial=(0.0,0.0), time_limit_s=5.0):
    """Global LP, continuous linear ramps; accepted only after exact path validation.

    No pre-t=0 actuation: f0=0 and empty command history, even for the oracle.
    A future disturbance IS known to this routine, allowing preaction after t=0.
    """
    start = time.perf_counter()
    n = model.steps(horizon_s)
    jumps, d = disturbance.arrays(model,n)
    delay, alpha = actuator.delay_steps(model.dt_s), actuator.alpha(model.dt_s)
    if len(initial) != 2 or not np.all(np.isfinite(initial)):
        raise ValueError('initial state must contain two finite SI quantities')
    # Variables [x0,v0,...,xN,vN, f0,...,fN, u0,...,uN-1, |u0|,...].
    fi, ui, ai = 2*(n+1), 3*(n+1), 3*(n+1)+n
    size = ai+n
    eq, ub = SparseRows(size), SparseRows(size)
    A, B0, B1, Bd = model.matrices()
    for k in range(n):
        for j in range(2):
            eq.add({2*(k+1)+j:1, 2*k:-A[j,0], 2*k+1:-A[j,1],
                    fi+k:-B0[j], fi+k+1:-B1[j]}, Bd[j]*d[k]+(j==1)*jumps[k+1])
        entries = {fi+k+1:1, fi+k:-(1-alpha)}
        if k >= delay:
            entries[ui+k-delay] = -alpha
        eq.add(entries, 0)
        ub.add({fi+k+1:1, fi+k:-1}, actuator.rate_N_s*model.dt_s)
        ub.add({fi+k+1:-1, fi+k:1}, actuator.rate_N_s*model.dt_s)
        ub.add({ui+k:1, ai+k:-1}, 0)
        ub.add({ui+k:-1, ai+k:-1}, 0)
        if jumps[k+1]:
            # Also bound the pre-jump endpoint velocity.
            ub.add({2*(k+1)+1:1}, model.v_abs_max_m_s+jumps[k+1])
            ub.add({2*(k+1)+1:-1}, model.v_abs_max_m_s-jumps[k+1])
    bounds = [b for _ in range(n+1) for b in [(-model.x_safe_m,model.x_safe_m),
                                                             (-model.v_abs_max_m_s,model.v_abs_max_m_s)]]
    bounds += [(-actuator.force_max_N,actuator.force_max_N)]*(n+1)
    bounds += [(-actuator.command_max_N,actuator.command_max_N)]*n
    bounds += [(0,actuator.command_max_N)]*n
    # Equality initial conditions must coexist with path bounds (not overwrite them).
    eq.add({0:1}, initial[0]); eq.add({1:1},initial[1]+jumps[0]); eq.add({fi:1},0)
    bounds[2*n] = (-model.x_terminal_m,model.x_terminal_m)
    bounds[2*n+1] = (-model.v_terminal_m_s,model.v_terminal_m_s)
    c = np.zeros(size)
    c[ai:] = model.dt_s/actuator.command_max_N  # soft L1 tie breaker; never a hard budget
    # Explicit dimensionless variable and row scaling; physical constraints stay identical.
    scales = np.array([b for _ in range(n+1) for b in (model.x_safe_m,model.v_abs_max_m_s)]
                      + [actuator.force_max_N]*(n+1)+[actuator.command_max_N]*(2*n))
    def scaled_rows(rows):
        matrix = rows.matrix() @ diags(scales)
        norms = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
        row_scales = 1/np.maximum(norms,1e-15)
        return diags(row_scales) @ matrix,np.asarray(rows.rhs)*row_scales
    eq_matrix, eq_rhs = scaled_rows(eq)
    ub_matrix, ub_rhs = scaled_rows(ub)
    scaled_bounds = [(lo/scale,hi/scale) for (lo,hi),scale in zip(bounds,scales)]
    solver_start = time.perf_counter()
    try:
        answer = linprog(c*scales,A_ub=ub_matrix,b_ub=ub_rhs,A_eq=eq_matrix,b_eq=eq_rhs,
                         bounds=scaled_bounds,method='highs-ipm',
                         options={'time_limit':time_limit_s,'primal_feasibility_tolerance':1e-8,
                                  'dual_feasibility_tolerance':1e-8})
    except Exception as exc:
        return Result(Status.SOLVER_ERROR.value,f'{type(exc).__name__}: {exc}',time.perf_counter()-start)
    solver_time = time.perf_counter()-solver_start
    elapsed = time.perf_counter()-start
    if answer.status == 2:
        return Result(Status.INFEASIBLE.value,answer.message,elapsed,solver_status=answer.status,solver_time_s=solver_time)
    if answer.status != 0 or answer.x is None:
        return Result(Status.SOLVER_ERROR.value,answer.message,elapsed,solver_status=answer.status,solver_time_s=solver_time)
    physical = answer.x*scales
    state = physical[:fi].reshape(n+1,2)
    force, command = physical[fi:ui],physical[ui:ai]
    try:
        metrics = validate(model,actuator,disturbance,state,force,command,initial)
        valid = metrics['valid_path_dynamics'] and metrics['terminal_met']
        reason = answer.message if valid else 'LP optimum failed independent posterior validation (unknown feasibility)'
    except Exception as exc:
        valid, metrics, reason = False, {}, f'validation error: {type(exc).__name__}: {exc}'
    return Result(Status.RECOVERABLE.value if valid else Status.SOLVER_ERROR.value,
                  reason,time.perf_counter()-start,state,force,command,metrics,answer.status,solver_time)
