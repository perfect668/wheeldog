"""Static joint-budget LP; point forces only, no gravity/stability guarantee."""
import numpy as np
from scipy.optimize import linprog

def ray_capacity(contact_map,budgets,direction,task=None,equal_increment=True):
    a=np.asarray(contact_map,float);budgets=np.asarray(budgets,float)
    d=np.asarray(direction,float)
    if not np.isfinite(a).all() or not np.isfinite(d).all() or abs(np.linalg.norm(d)-1)>1e-10:
        raise ValueError('Nonfinite map or non-unit direction')
    if np.any(budgets<0) or np.any(np.isnan(budgets)):raise ValueError('Invalid budget')
    f0=np.zeros(6) if task is None else np.asarray(task,float)
    if f0.shape!=(6,) or not np.isfinite(f0).all():raise ValueError('Bad task')
    mask=np.isfinite(budgets)
    # Force increment is allocated jointly. Main model equal increments; optional
    # free split is a distinct ideal contact relaxation (not used to pair poses).
    eq=np.zeros((6 if equal_increment else 3,7));eq[:3,:3]=np.eye(3);eq[:3,3:6]=np.eye(3);eq[:3,6]=-d
    if equal_increment:eq[3:,:3]=np.eye(3);eq[3:,3:6]=-np.eye(3)
    ub=np.vstack([np.c_[a[mask],np.zeros(mask.sum())],np.c_[-a[mask],np.zeros(mask.sum())]])
    rhs=np.r_[budgets[mask]-a[mask]@f0,budgets[mask]+a[mask]@f0]
    if np.any(rhs<-1e-7):return dict(status='TASK_INFEASIBLE',capacity_n=None,reason='task exceeds nominal torque budget')
    opt=linprog(np.r_[np.zeros(6),-1.],A_ub=ub,b_ub=rhs,A_eq=eq,b_eq=np.zeros(len(eq)),
                bounds=[(None,None)]*6+[(0,None)],method='highs')
    status={0:'OPTIMAL',2:'LP_INFEASIBLE',3:'UNBOUNDED'}.get(opt.status,'SOLVER_ERROR')
    result=dict(status=status,solver_status=int(opt.status),message=opt.message,capacity_n=None)
    if opt.success:
        f=f0+opt.x[:6];tau=a@f;violation=max(0.,float(np.max(abs(tau[mask])-budgets[mask])))
        equality=float(np.max(abs(eq@opt.x)))
        if violation>1e-6 or equality>1e-6 or not np.isfinite(opt.x).all():
            result.update(status='SOLVER_ERROR',reason='posterior LP violation');return result
        ratios=abs(tau[mask])/np.maximum(budgets[mask],1e-12)
        result.update(capacity_n=float(opt.x[-1]),forces_robot_on_human_n=f.tolist(),
            compensating_torque_nm=tau.tolist(),limiting_dofs=np.flatnonzero(mask)[ratios>=1-1e-6].tolist(),
            max_violation_nm=violation,max_equality_error_n=equality)
    return result
