"""Same offline LP for every posture; known disturbances, spatial-chart class."""
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import diags
from recoverability.solver import SparseRows
from scipy.signal import place_poles


def state_bounds(model,envelope):
    h=model.human;e=envelope
    return np.array([h.x_safe_m,h.v_abs_max_m_s,max(abs(e['y_lower_m']),abs(e['y_upper_m'])),e['speed_m_s']])


def solve_offline(model,disturbance,horizon,envelope,delay_s,time_limit=5.):
    start=time.perf_counter();n=model.human.steps(horizon);jumps,d=disturbance.arrays(model.human,n)
    D=int(round(delay_s/model.dt))
    if abs(D*model.dt-delay_s)>1e-10:raise ValueError('delay must be an integer number of intervals')
    ui=4*(n+1);ai=ui+n;size=ai+n;eq=SparseRows(size);ub=SparseRows(size)
    A,B,Bd=model.matrices();force=model.force_row
    for k in range(n):
        for j in range(4):
            row={4*(k+1)+j:1,ui+k:-B[j]}
            row.update({4*k+i:-A[j,i] for i in range(4)})
            eq.add(row,Bd[j]*d[k]+(j==1)*jumps[k+1])
        ub.add({ui+k:1,ai+k:-1},0);ub.add({ui+k:-1,ai+k:-1},0)
        # Exact linear flow maps at quarter points constrain contact and human
        # states between knots; denser nonlinear posterior checks remain required.
        for t in [0.,model.dt*.25,model.dt*.5,model.dt*.75,model.dt]:
            At,Bt,Dt=model.matrices(t)
            transforms=[(At[0],Bt[0],Dt[0],-model.human.x_safe_m,model.human.x_safe_m),
                (At[1],Bt[1],Dt[1],-model.human.v_abs_max_m_s,model.human.v_abs_max_m_s),
                (force@At,float(force@Bt),float(force@Dt),envelope['per_hand_force_lower_n'],envelope['per_hand_force_upper_n'])]
            for coeff,bu,bd,lo,hi in transforms:
                row={4*k+i:coeff[i] for i in range(4)};row[ui+k]=bu
                ub.add(row,hi-bd*d[k]);ub.add({i:-v for i,v in row.items()},-lo+bd*d[k])
    bounds=[]
    # A quadratic y segment differs from its endpoint chord by at most A*dt^2/8.
    # Inward node margin therefore bounds its whole continuous spatial domain.
    domain_margin=envelope['acceleration_m_s2']*model.dt**2/8+1e-10
    for k in range(n+1):
        bounds += [(-model.human.x_safe_m,model.human.x_safe_m),(-model.human.v_abs_max_m_s,model.human.v_abs_max_m_s),
            (envelope['y_lower_m']+domain_margin,envelope['y_upper_m']-domain_margin),(-envelope['speed_m_s'],envelope['speed_m_s'])]
    initial=np.array([0.,jumps[0],0,0])
    for j in range(4):eq.add({j:1},initial[j]);eq.add({4*n+j:1},0.)
    amax=envelope['acceleration_m_s2']
    bounds += [(0.,0.) if k<D else (-amax,amax) for k in range(n)]+[(0,amax)]*n
    scales=np.r_[np.tile(state_bounds(model,envelope),n+1),np.full(2*n,amax)]
    def scale_rows(rows):
        matrix=rows.matrix()@diags(scales);norm=np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
        sc=1/np.maximum(norm,1e-12)
        return diags(sc)@matrix,np.array(rows.rhs)*sc
    ae,be=scale_rows(eq);au,bu=scale_rows(ub)
    cost=np.zeros(size);cost[ai:]=model.dt/max(amax,1e-12)
    try:
        opt=linprog(cost*scales,A_ub=au,b_ub=bu,A_eq=ae,b_eq=be,
            bounds=[(lo/s,hi/s) for (lo,hi),s in zip(bounds,scales)],method='highs-ipm',
            options={'time_limit':time_limit,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    except Exception as exc:
        return dict(status='SOLVER_ERROR',reason=f'{type(exc).__name__}: {exc}',elapsed_s=time.perf_counter()-start)
    result=dict(status={0:'LP_FEASIBLE',2:'RESTRICTED_LP_INFEASIBLE'}.get(opt.status,'SOLVER_ERROR'),
        solver_status=int(opt.status),reason=opt.message,elapsed_s=time.perf_counter()-start,
        information='offline known complete disturbance; infeasibility only within specified chart/envelope/control class')
    if opt.success:
        z=opt.x*scales;state=z[:ui].reshape(n+1,4);acceleration=z[ui:ai]
        result.update(state=state,acceleration=acceleration,objective=float(opt.fun),
            max_equality_scaled_error=float(np.max(abs(ae@opt.x-be))),
            max_inequality_scaled_violation=float(max(0,np.max(au@opt.x-bu))))
    return result


class CausalFeedback:
    """Fixed pole-placement feedback and scalar constraint governor, not MPC.

    API receives only current observed state and its own delivered command queue.
    Pending accelerations are propagated with zero unknown future disturbance.
    """
    def __init__(self,model,envelope,delay_s,poles):
        self.model=model;self.envelope=envelope
        self.gain=place_poles(model.M,model.u[:,None],poles).gain_matrix[0]
        D=int(round(delay_s/model.dt))
        if abs(D*model.dt-delay_s)>1e-10:raise ValueError('nongrid delay')
        self.queue=[0.]*D;self.raw=[];self.sent=[]

    def command(self,observed):
        s=np.array(observed,float);m=self.model;e=self.envelope
        for a in self.queue:s=m.step(s,a)
        raw=-float(self.gain@s);lo=-e['acceleration_m_s2'];hi=-lo
        # Include exact w and quadratic y interior extrema in posterior checks.
        for t in [m.dt*.25,m.dt*.5,m.dt*.75,m.dt]:
            A,B,_=m.matrices(t);pred=A@s
            for row,lower,upper in [(np.array([0,0,1.,0]),e['y_lower_m'],e['y_upper_m']),
                    (np.array([0,0,0,1.]),-e['speed_m_s'],e['speed_m_s']),
                    (m.force_row,e['per_hand_force_lower_n'],e['per_hand_force_upper_n'])]:
                slope=float(row@B);offset=float(row@pred)
                if abs(slope)<1e-12:
                    if offset<lower-1e-7 or offset>upper+1e-7:return None,dict(reason='governor_no_admissible_acceleration',raw=raw)
                else:
                    a,b=sorted([(lower-offset)/slope,(upper-offset)/slope]);lo=max(lo,a);hi=min(hi,b)
        if lo>hi+1e-8:return None,dict(reason='governor_no_admissible_acceleration',raw=raw,lower=lo,upper=hi)
        sent=float(np.clip(raw,lo,hi));self.sent.append(sent);self.raw.append(raw)
        if self.queue:
            delivered=self.queue.pop(0);self.queue.append(sent)
        else:delivered=sent
        return delivered,dict(raw=raw,sent=sent,delivered=delivered,allowed_lower=lo,allowed_upper=hi)


def simulate_feedback(model,disturbance,horizon,envelope,delay_s,poles):
    start=time.perf_counter();n=model.human.steps(horizon);jumps,d=disturbance.arrays(model.human,n)
    state=np.zeros((n+1,4));state[0,1]=jumps[0];controls=[];logs=[]
    controller=CausalFeedback(model,envelope,delay_s,poles);reason='completed';failure=None
    for k in range(n):
        a,log=controller.command(state[k]);logs.append(log)
        if a is None:reason=log['reason'];failure=k;break
        controls.append(a);state[k+1]=model.step(state[k],a,d[k]);state[k+1,1]+=jumps[k+1]
        if abs(state[k+1,0])>model.human.x_safe_m+1e-7 or abs(state[k+1,1])>model.human.v_abs_max_m_s+1e-7:
            reason='human_path_domain_exceeded';failure=k+1;break
    length=len(controls)
    return dict(status='SIMULATED' if length==n else 'FEEDBACK_CONSTRAINT_FAILURE',reason=reason,
        state=state[:length+1],acceleration=np.array(controls),command_logs=logs,
        first_failure_node=failure,elapsed_s=time.perf_counter()-start,
        information='fixed linear state feedback, current exact state and own queue only; failure is not infeasibility')
