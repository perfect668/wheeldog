"""Causal finite-horizon QP, observed state and own pending actions only.

The four-state contact dynamics, exact integration, and fallback governor are
imported from M1-D. The robot follows a precomputed admissible spatial curve.
No future event, case identifier, scoring trajectory, or recovery deadline is
an argument to either command interface.
"""
import time
import numpy as np
import casadi as ca
from scipy.linalg import solve_discrete_are
from scipy.sparse import lil_matrix,csc_matrix,block_diag,vstack
from h1_coupled.solver import CausalFeedback


class PredictiveController:
    def __init__(self,model,envelope,delay_s,options):
        self.model=model;self.envelope=envelope;self.options=options
        self.fallback=CausalFeedback(model,envelope,delay_s,options['poles_s_inv'])
        D=int(round(delay_s/model.dt))
        if abs(D*model.dt-delay_s)>1e-10:raise ValueError('nongrid delay')
        self.queue=[0.]*D;self.n=int(round(options['mpc_horizon_s']/model.dt));self.warm=None
        self.scale=np.array(options['state_scale'],float);T=np.diag(self.scale);Ti=np.diag(1/self.scale)
        A,B,_=model.matrices();self.A=Ti@A@T;self.B=Ti@B
        Q=np.diag(options['state_weights']);R=options['acceleration_weight']
        P=solve_discrete_are(self.A,self.B[:,None],Q,[[R]])
        n=self.n;ui=4*(n+1);self.ui=ui;self.size=ui+n
        H=block_diag([Q]*n+[P]+[csc_matrix([[R]])]*n,format='csc')*2
        E=lil_matrix((4*(n+1),self.size));E[:4,:4]=np.eye(4)
        for k in range(n):
            E[4*(k+1):4*(k+2),4*(k+1):4*(k+2)]=np.eye(4)
            E[4*(k+1):4*(k+2),4*k:4*(k+1)]=-self.A
            E[4*(k+1):4*(k+2),ui+k]=-self.B[:,None]
        C=[];lo=[];hi=[];e=envelope
        # Normalize constraint rows by their physical state/force scales.
        limits=[(-model.human.x_safe_m,model.human.x_safe_m),(-model.human.v_abs_max_m_s,model.human.v_abs_max_m_s),
            (e['y_lower_m'],e['y_upper_m']),(-e['speed_m_s'],e['speed_m_s'])]
        for k in range(n+1):
            for axis,(lower,upper) in enumerate(limits):
                row=np.zeros(self.size);row[4*k+axis]=1
                # Node inset ensures quadratic handle motion remains in domain.
                margin=e['acceleration_m_s2']*model.dt**2/8 if axis==2 and k>0 else 0
                C.append(row);lo.append((lower+margin)/self.scale[axis]);hi.append((upper-margin)/self.scale[axis])
        for k in range(n):
            for fraction in [.25,.5,.75,1.]:
                At,Bt,_=model.matrices(model.dt*fraction)
                f=model.force_row@At@T/model.cap_y;bu=float(model.force_row@Bt)/model.cap_y
                row=np.zeros(self.size);row[4*k:4*(k+1)]=f;row[ui+k]=bu
                C.append(row);lo.append(e['per_hand_force_lower_n']/model.cap_y);hi.append(e['per_hand_force_upper_n']/model.cap_y)
            row=np.zeros(self.size);row[ui+k]=1
            C.append(row);lo.append(-e['acceleration_m_s2']);hi.append(e['acceleration_m_s2'])
        C=csc_matrix(np.array(C));self.G=vstack([E.tocsc(),C],format='csc')
        self.lower=np.r_[np.zeros(E.shape[0]),lo];self.upper=np.r_[np.zeros(E.shape[0]),hi]
        def dm(s):
            s=csc_matrix(s);s.sort_indices()
            return ca.DM(ca.Sparsity(s.shape[0],s.shape[1],s.indptr.tolist(),s.indices.tolist()),s.data)
        self.gdm=dm(self.G);self.hdm=dm(H)
        self.solver=ca.conic('mpc','osqp',{'h':self.hdm.sparsity(),'a':self.gdm.sparsity()},
            {'error_on_fail':False,'print_time':False,'osqp':dict(verbose=False,eps_abs=options['qp_eps_abs'],
             eps_rel=options['qp_eps_rel'],max_iter=options['qp_max_iter'],polish=True)})

    def command(self,observed):
        started=time.perf_counter();s=np.array(observed,float);m=self.model
        # Unknown disturbances are zero in this prediction. Committed controls
        # cannot be changed after an event arrives.
        pending_violation=False
        for a in self.queue:
            for t in [.25*m.dt,.5*m.dt,.75*m.dt,m.dt]:
                p=m.step(s,a,duration=t);e=self.envelope
                pending_violation |= bool(abs(p[0])>m.human.x_safe_m+1e-7 or abs(p[1])>m.human.v_abs_max_m_s+1e-7
                    or p[2]<e['y_lower_m']-1e-7 or p[2]>e['y_upper_m']+1e-7 or abs(p[3])>e['speed_m_s']+1e-7
                    or m.per_hand_force(p)<e['per_hand_force_lower_n']-1e-7 or m.per_hand_force(p)>e['per_hand_force_upper_n']+1e-7)
            s=m.step(s,a)
        lower=self.lower.copy();upper=self.upper.copy();lower[:4]=s/self.scale;upper[:4]=lower[:4]
        log={};sent=None
        try:
            args=dict(h=self.hdm,g=np.zeros(self.size),a=self.gdm,lba=lower,uba=upper)
            if self.warm is not None:args['x0']=self.warm
            result=self.solver(**args);stats=self.solver.stats();z=np.array(result['x']).ravel()
            residual=float(max(0,np.max(lower-self.G@z),np.max(self.G@z-upper))) if np.isfinite(z).all() else np.inf
            log=dict(qp_status=stats.get('return_status'),qp_success=bool(stats.get('success')),qp_iterations=stats.get('iter_count'),
                qp_max_normalized_violation=residual if np.isfinite(residual) else None)
            if stats.get('success') and residual<=self.options['acceptance_abs_residual']:
                sent=float(z[self.ui]);self.warm=z
        except Exception as exc:log=dict(qp_status='EXCEPTION',exception=f'{type(exc).__name__}: {exc}')
        log['pending_queue_constraint_violation']=pending_violation
        log['fallback']=sent is None
        if sent is None:
            # The fallback has the identical pending queue; it must not invent a
            # different delivered acceleration to hide a QP failure.
            self.fallback.queue=self.queue.copy();delivered,base=self.fallback.command(observed)
            self.queue=self.fallback.queue.copy();log['governor']=base
            if delivered is None:log['reason']='qp_and_governor_failure'
        elif self.queue:
            delivered=self.queue.pop(0);self.queue.append(sent)
        else:delivered=sent
        log['sent']=sent if sent is not None else log.get('governor',{}).get('sent')
        log['delivered']=delivered;log['wall_s']=time.perf_counter()-started
        # Bundled CasADi OSQP 0.6.3 exposes an iteration budget but no interruptible
        # wall-time limit. Timing targets are diagnostics, not hidden deadlines.
        log['above_timing_target']=log['wall_s']>self.options['qp_timing_target_s']
        log['missed_control_period']=log['wall_s']>m.dt
        return delivered,log


def make_controller(name,model,envelope,delay,options):
    if name=='mpc':return PredictiveController(model,envelope,delay,options)
    if name=='fixed':return CausalFeedback(model,envelope,delay,options['poles_s_inv'])
    raise ValueError(name)
