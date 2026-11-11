"""Coupled tangent velocity LP: both handles translate together, no rotation."""
import numpy as np
from scipy.optimize import linprog

def cooperative_velocity(model,q,base,direction,profile,mobile):
    j=model.task_jacobian(q,base,mobile);n=j.shape[1];d=np.r_[direction,[0,0,0]]
    eq=np.c_[j,-np.tile(d,2)]
    vmax=model.coordinate_velocity(profile,mobile)
    lo=-vmax.copy();hi=vmax.copy()
    for k,i in enumerate(model.active):
        if q[i]<=model.robot.lower[i]+1e-8:lo[k]=0
        if q[i]>=model.robot.upper[i]-1e-8:hi[k]=0
    aub=None;bub=None
    if mobile:
        # Inscribed 32-gon speed disk: polygon vertices lie at v_max.
        theta=(np.arange(32)+.5)*2*np.pi/32
        aub=np.zeros((32,n+1));aub[:,-4]=np.cos(theta);aub[:,-3]=np.sin(theta)
        bub=np.full(32,model.config['base_velocity_m_s']*np.cos(np.pi/32))
    opt=linprog(np.r_[np.zeros(n),-1.],A_eq=eq,b_eq=np.zeros(12),A_ub=aub,b_ub=bub,
                bounds=list(zip(lo,hi))+[(0,None)],method='highs')
    result=dict(status={0:'OPTIMAL',2:'LP_INFEASIBLE',3:'UNBOUNDED'}.get(opt.status,'SOLVER_ERROR'),
                solver_status=int(opt.status),message=opt.message,speed_m_s=None)
    if opt.success:
        if np.max(abs(eq@opt.x))>1e-7:result['status']='SOLVER_ERROR';return result
        singular=np.linalg.svd(j,compute_uv=False)
        result.update(speed_m_s=float(opt.x[-1]),coordinate_velocity=opt.x[:-1].tolist(),
            min_task_singular_value=float(singular[-1]),rank=int(np.linalg.matrix_rank(j,tol=1e-8)))
    return result
