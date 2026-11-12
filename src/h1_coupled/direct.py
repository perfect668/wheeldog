"""Independent direct joint-acceleration trajectory optimization.

Sequential linearization of FK/contact dynamics with a trust region. Joint
positions/velocities/accelerations at all knots are optimized; no prescribed
minimum-jerk Cartesian path or spatial IK curve is used. This local method has
no global infeasibility semantics. Every accepted result is replayed with true
FK/contact forces and an independent human ODE.
"""
from pathlib import Path
import argparse
import json
import time
import yaml
import subprocess
import concurrent.futures
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from recoverability.solver import SparseRows
from h1_capability.reachability import pose_error
from h1_kinematics.assets import dump
from .contact import CoupledModel
from .experiment import load_config,init_worker,get_chart


def optimize_iteration(chart,model,reference,horizon,sign,config,trust=.15):
    robot=chart.model;nj=len(robot.active);ns=2+2*nj;dt=model.dt;n=model.human.steps(horizon)
    ui=ns*(n+1);pi=ui+nj*n;ai=pi+1;size=ai+nj*n
    eq=SparseRows(size);ub=SparseRows(size);bounds=[]
    force_rows=[];dynamics=[];q0=chart.q0[robot.active];budget=robot.budgets().copy()
    budget[robot.torso_indices]=config['robot']['torso_budget_nm'];mask=np.isfinite(budget)
    for k in range(n+1):
        ref=reference[k];q=chart.q0.copy();q[robot.active]=q0+ref
        poses=robot.poses(q,chart.b0);jac=robot.task_jacobian(q,chart.b0,False)
        errors=pose_error(poses,chart.initial).reshape(2,6)
        fs=[];offsets=[]
        for side in range(2):
            jy=jac[6*side+1]
            f=np.r_[-model.k,-model.c,model.k*jy,model.c*jy]
            offset=model.k*(errors[side,1]-jy@ref)
            fs.append(f);offsets.append(offset)
            # World x,z and all three orientation components remain fixed.
            for axis in [0,2,3,4,5]:
                row={ns*k+2+j:jac[6*side+axis,j] for j in range(nj)}
                target=float(jac[6*side+axis]@ref-errors[side,axis])
                tolerance=(config['robot']['position_tolerance_m'] if axis<3 else config['robot']['rotation_tolerance_rad'])/5
                ub.add(row,target+tolerance);ub.add({i:-v for i,v in row.items()},-target+tolerance)
        row={ns*k+2+j:jac[1,j]-jac[7,j] for j in range(nj)}
        target=float((jac[1]-jac[7])@ref-errors[0,1]+errors[1,1]);tolerance=config['robot']['position_tolerance_m']/5
        ub.add(row,target+tolerance);ub.add({i:-v for i,v in row.items()},-target+tolerance)
        # Constrain actual quadratic joint segments inside each control interval.
        # Knot-only orientation constraints can miss inter-knot rotation errors.
        if k<n:
            for fraction in config['direct'].get('geometry_subintervals',[]):
                t=fraction*dt;ref_mid=(1-fraction)*reference[k]+fraction*reference[k+1]
                q_mid=chart.q0.copy();q_mid[robot.active]=q0+ref_mid
                j_mid=robot.task_jacobian(q_mid,chart.b0,False)
                err_mid=pose_error(robot.poses(q_mid,chart.b0),chart.initial).reshape(2,6)
                specifications=[(j_mid[6*s+axis],err_mid[s,axis],axis) for s in range(2) for axis in [0,2,3,4,5]]
                specifications.append((j_mid[1]-j_mid[7],err_mid[0,1]-err_mid[1,1],1))
                for jac_row,error,axis in specifications:
                    row={ns*k+2+j:jac_row[j] for j in range(nj)}
                    row.update({ns*k+2+nj+j:jac_row[j]*t for j in range(nj)})
                    row.update({ui+nj*k+j:jac_row[j]*.5*t*t for j in range(nj)})
                    target=float(jac_row@ref_mid-error)
                    tolerance=(config['robot']['position_tolerance_m'] if axis<3 else config['robot']['rotation_tolerance_rad'])/5
                    ub.add(row,target+tolerance);ub.add({i:-v for i,v in row.items()},-target+tolerance)
        force_rows.append((fs,offsets))
        for f,offset in zip(fs,offsets):
            ub.add({ns*k+i:v for i,v in enumerate(f)},model.cap_y-offset)
            ub.add({ns*k+i:-v for i,v in enumerate(f)},model.cap_y+offset)
        contact=robot.contact_map(q,chart.b0)
        vertical=contact@np.tile([0.,0,model.fz],2)
        for i in np.flatnonzero(mask):
            coeff=contact[i,1]*fs[0]+contact[i,4]*fs[1]
            constant=vertical[i]+contact[i,1]*offsets[0]+contact[i,4]*offsets[1]
            ub.add({ns*k+j:v for j,v in enumerate(coeff)},budget[i]-constant)
            ub.add({ns*k+j:-v for j,v in enumerate(coeff)},budget[i]+constant)
        bounds += [(-model.human.x_safe_m,model.human.x_safe_m),(-model.human.v_abs_max_m_s,model.human.v_abs_max_m_s)]
        bounds += [(max(robot.robot.lower[idx]-q0[j],ref[j]-trust),min(robot.robot.upper[idx]-q0[j],ref[j]+trust)) for j,idx in enumerate(robot.active)]
        bounds += [(-v,v) for v in robot.velocity]
        if k==n:continue
        M=np.zeros((ns,ns));M[0,1]=1;M[1,0]=model.human.omega**2
        M[1]+=(fs[0]+fs[1])/model.human.mass_kg
        M[2:2+nj,2+nj:]=np.eye(nj)
        U=np.zeros((ns,nj));U[2+nj:]=np.eye(nj)
        const=np.zeros(ns);const[1]=sum(offsets)/model.human.mass_kg
        aug=np.zeros((ns+nj+1,ns+nj+1));aug[:ns,:ns]=M;aug[:ns,ns:ns+nj]=U;aug[:ns,-1]=const
        E=expm(aug*dt);A=E[:ns,:ns];B=E[:ns,ns:ns+nj];C=E[:ns,-1];dynamics.append((A,B,C))
        for j in range(ns):
            row={ns*(k+1)+j:1};row.update({ns*k+i:-A[j,i] for i in range(ns)})
            row.update({ui+nj*k+i:-B[j,i] for i in range(nj)})
            eq.add(row,C[j])
    for j in range(ns):
        row={j:1}
        if j==1:row[pi]=-sign
        eq.add(row,0);eq.add({ns*n+j:1},0)
    amax=config['_profile']['joint_acceleration_rad_s2'];D=int(round(config['_profile']['delay_s']/dt))
    bounds += [(0,0) if k<D else (-amax,amax) for k in range(n) for _ in range(nj)]
    bounds += [(0,config['direct']['impulse_cap_m_s'])]+[(0,amax)]*(nj*n)
    for j in range(nj*n):
        ub.add({ui+j:1,ai+j:-1},0);ub.add({ui+j:-1,ai+j:-1},0)
    scales=np.r_[np.tile(np.r_[.2,2.,np.ones(nj)*.3,robot.velocity],n+1),np.full(nj*n,amax),.5,np.full(nj*n,amax)]
    def scaled(rows):
        mat=rows.matrix()@diags(scales);norm=np.asarray(abs(mat).max(axis=1).toarray()).ravel();s=1/np.maximum(norm,1e-12)
        return diags(s)@mat,np.array(rows.rhs)*s
    ae,be=scaled(eq);au,bu=scaled(ub)
    cost=np.zeros(size);cost[pi]=-1;cost[ai:]=1e-7*dt
    opt=linprog(cost*scales,A_eq=ae,b_eq=be,A_ub=au,b_ub=bu,bounds=[(l/s,h/s) for (l,h),s in zip(bounds,scales)],
        method='highs-ipm',options={'time_limit':config['experiment']['lp_time_limit_s']*4,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    result=dict(status={0:'LINEARIZED_OPTIMUM',2:'LOCAL_SUBPROBLEM_INFEASIBLE'}.get(opt.status,'SOLVER_ERROR'),solver_status=int(opt.status),message=opt.message)
    if opt.success:
        z=opt.x*scales;result.update(states=z[:ui].reshape(n+1,ns),acceleration=z[ui:pi].reshape(n,nj),impulse_m_s=float(z[pi]))
    return result


def replay(chart,model,states,acceleration,impulse,sign,config,save_path=None):
    robot=chart.model;n=len(acceleration);nj=len(robot.active);dt=model.dt
    initial=chart.initial;human=np.array([0.,sign*impulse]);humans=[human.copy()];trace=[];reasons=[]
    sub=max(2,int(np.ceil(dt/config['experiment']['validation_dt_s'])))
    peakv=peaka=force_ratio=torque_ratio=0.;pe=re=0.;minimum=np.inf
    budgets=robot.budgets().copy();budgets[robot.torso_indices]=config['robot']['torso_budget_nm'];mask=np.isfinite(budgets)
    def mechanical(k,t):
        s=states[k];q=chart.q0.copy();q[robot.active]+=s[2:2+nj]+s[2+nj:]*t+.5*acceleration[k]*t*t
        v=s[2+nj:]+acceleration[k]*t;poses=robot.poses(q,chart.b0);jac=robot.task_jacobian(q,chart.b0,False)
        hy=np.array([p[1,3]-p0[1,3] for p,p0 in zip(poses,initial)]);hv=np.array([jac[1]@v,jac[7]@v])
        return q,v,poses,hy,hv
    for k in range(n):
        times=np.linspace(0,dt,sub+1)
        def rhs(t,h):
            _,_,_,hy,hv=mechanical(k,t);force=model.k*(hy-h[0])+model.c*(hv-h[1])
            return [h[1],model.human.omega**2*h[0]+sum(force)/model.human.mass_kg]
        sol=solve_ivp(rhs,(0,dt),human,t_eval=times,method='DOP853',rtol=2e-10,atol=2e-12)
        if not sol.success:reasons.append('independent_ode_failure');break
        for j,t in enumerate(times):
            h=sol.y[:,j];q,v,poses,hy,hv=mechanical(k,t);force=model.k*(hy-h[0])+model.c*(hv-h[1])
            error=pose_error(poses,initial).reshape(2,6)
            # Lateral common motion is free, all other components fixed.
            err=error.copy();err[:,1]-=np.mean(error[:,1])
            pe=max(pe,float(max(np.linalg.norm(err[:,:3],axis=1))));re=max(re,float(max(np.linalg.norm(err[:,3:],axis=1))))
            tau=robot.contact_map(q,chart.b0)@np.r_[0.,force[0],model.fz,0.,force[1],model.fz]
            force_ratio=max(force_ratio,float(max(np.hypot(force,model.fz))/model.contact['resultant_cap_per_hand_n']))
            torque_ratio=max(torque_ratio,float(max(abs(tau[mask])/budgets[mask])))
            peakv=max(peakv,float(max(abs(v)/robot.velocity)));peaka=max(peaka,float(max(abs(acceleration[k]))/config['_profile']['joint_acceleration_rad_s2']))
            if np.any(q<robot.robot.lower-1e-8) or np.any(q>robot.robot.upper+1e-8):reasons.append('joint_position')
            # Acceleration is piecewise constant; position extrema may be interior.
            for coordinate in range(nj):
                aa=acceleration[k,coordinate]
                if abs(aa)>1e-12:
                    critical=-states[k,2+nj+coordinate]/aa
                    if 0<critical<dt:
                        value=chart.q0[robot.active[coordinate]]+states[k,2+coordinate]+states[k,2+nj+coordinate]*critical+.5*aa*critical**2
                        if value<robot.robot.lower[robot.active[coordinate]]-1e-8 or value>robot.robot.upper[robot.active[coordinate]]+1e-8:reasons.append('continuous_joint_position')
            if j%config['experiment']['collision_stride']==0 or j==sub:
                check=chart.checker.check(robot.robot.fk(q,chart.b0),config['robot']['clearance_m']);minimum=min(minimum,check['min_distance_m'])
                if not check['passed']:reasons.append('sampled_collision')
            trace.append((k*dt+t,h.copy(),q,v.copy(),acceleration[k].copy(),hy,hv,force,tau))
        human=sol.y[:,-1];humans.append(human.copy())
    if not trace:return dict(valid=False,reasons=['no_trace'])
    if pe>config['robot']['position_tolerance_m'] or re>config['robot']['rotation_tolerance_rad']:reasons.append('handle_coordination')
    if force_ratio>1+2e-6:reasons.append('contact_resultant_cap')
    if torque_ratio>1+2e-6:reasons.append('contact_torque_budget')
    if max(peakv,peaka)>1+2e-6:reasons.append('joint_velocity_or_acceleration')
    hs=np.array([r[1] for r in trace])
    if max(abs(hs[:,0]))>model.human.x_safe_m+1e-7:reasons.append('human_path_position')
    if max(abs(hs[:,1]))>model.human.v_abs_max_m_s+1e-7:reasons.append('human_path_velocity')
    if abs(human[0])>model.human.x_terminal_m+1e-7 or abs(human[1])>model.human.v_terminal_m_s+1e-7:reasons.append('human_terminal')
    terminal_joint=max(abs(states[-1,2:]))
    if terminal_joint>1e-7:reasons.append('robot_terminal')
    metrics=dict(valid=not reasons,reasons=sorted(set(reasons)),terminal_human=human.tolist(),
        max_handle_position_error_m=pe,max_handle_rotation_error_rad=re,max_contact_cap_ratio=force_ratio,
        max_torque_ratio=torque_ratio,max_joint_velocity_ratio=peakv,max_joint_acceleration_ratio=peaka,
        min_sampled_clearance_m=minimum,peak_human_x_m=float(max(abs(hs[:,0]))),
        terminal_linearization_error_m_m_s=(human-states[-1,:2]).tolist(),
        interpretation='Local sequential joint trajectory optimization; actual nonlinear geometry/contact replay, sampled collision')
    if save_path is not None:
        np.savez_compressed(save_path,time_s=np.array([r[0] for r in trace]),human_state=hs,
            q_rad=np.array([r[2] for r in trace]),joint_velocity=np.array([r[3] for r in trace]),joint_acceleration=np.array([r[4] for r in trace]),
            handle_y_m=np.array([r[5] for r in trace]),handle_v_m_s=np.array([r[6] for r in trace]),per_hand_lateral_force_n=np.array([r[7] for r in trace]),
            torque_nm=np.array([r[8] for r in trace]),node_states=states,acceleration=acceleration,impulse_m_s=np.array(impulse),sign=np.array(sign))
    return metrics


def optimize_case(candidate,sign,c,out):
    init_worker(c,out);_,chart,_=get_chart(candidate,'fixed');dt=c['experiment']['dt_s'];model=CoupledModel(c['_human'],c['contact'],dt)
    h=c['direct']['horizon_s'];n=model.human.steps(h);nj=len(chart.model.active);attempts=[];best=None;started=time.perf_counter()
    main=json.loads((out/'main.json').read_text())
    nominal=[r for r in main if r['candidate']==candidate and r['mode']=='fixed' and r['method']=='offline'
        and r['horizon_s']==h and r['status']=='RECOVERED_OFFLINE' and sign*r['disturbance']['delta_v_m_s']>=0]
    seed=max(nominal,key=lambda r:abs(r['disturbance']['delta_v_m_s']),default=None)
    references=[]
    if seed:
        with np.load(seed['path_file'],allow_pickle=False) as data:
            y=data['node_state'][:,2]
        references.append(('chart_seed',chart.spline(y)[:,:nj]-chart.q0[chart.model.active]))
    references.append(('zero_seed',np.zeros((n+1,nj))))
    for label,ref in references:
        for iteration in range(c['direct']['sequential_iterations']):
            result=optimize_iteration(chart,model,ref,h,sign,c,trust=c['direct']['trust_region_rad'])
            log={k:v for k,v in result.items() if k not in ['states','acceleration']};log.update(start=label,iteration=iteration)
            if result['status']!='LINEARIZED_OPTIMUM':attempts.append(log);break
            states=result['states'];positions=states[:,2:2+nj];step=float(np.max(abs(positions-ref)));log['max_joint_iterate_change_rad']=step
            prefix=c['direct'].get('label','direct')
            raw=out/'direct_paths'/f'{prefix}_iterate_{candidate}_{sign:+d}_{label}_{iteration:02d}.npz'
            np.savez_compressed(raw,states=states,acceleration=result['acceleration'],impulse_m_s=result['impulse_m_s'],reference=ref)
            log['iterate_file']=str(raw)
            metrics=replay(chart,model,states,result['acceleration'],result['impulse_m_s'],sign,c)
            log['validation']=metrics;attempts.append(log)
            if metrics['valid'] and (best is None or result['impulse_m_s']>best['impulse_m_s']):
                best=dict(result,metrics=metrics,start=label,iteration=iteration)
            ref=positions.copy()
            if metrics['valid'] and step<c['direct']['convergence_rad']:break
    caseid=f"{c['direct'].get('label','direct')}_{candidate}_{sign:+d}";path=out/'direct_paths'/(caseid+'.npz')
    if best:
        replay(chart,model,best['states'],best['acceleration'],best['impulse_m_s'],sign,c,path)
    return dict(case_id=caseid,candidate=candidate,sign=sign,horizon_s=h,mode='fixed',
        status='VERIFIED_DIRECT_RECOVERY' if best else 'NO_VERIFIED_DIRECT_RECOVERY',
        achieved_impulse_m_s=best['impulse_m_s'] if best else None,metrics=best['metrics'] if best else None,
        attempts=attempts,path_file=str(path) if best else None,elapsed_s=time.perf_counter()-started,
        interpretation='Feasible local lower bound only; failed/rejected LP iterations do not prove global infeasibility')


def run(output,workers=2,override=None):
    out=Path(output);c=yaml.safe_load((out/'config.yaml').read_text());(out/'direct_paths').mkdir(exist_ok=True)
    if override:c['direct'].update(yaml.safe_load(Path(override).read_text()))
    prefix=c['direct'].get('label','direct')
    saved_config=out/(prefix+'_config.yaml')
    if saved_config.exists() and yaml.safe_load(saved_config.read_text())!=c:raise ValueError('direct configuration differs from saved run')
    saved_config.write_text(yaml.safe_dump(c,sort_keys=False))
    dump(out/(prefix+'_provenance.json'),dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(),
        started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(optimize_case,candidate,sign,c,out) for candidate in c['direct']['candidates'] for sign in c['direct']['signs']]
        for future in futures:
            r=future.result();dump(out/(r['case_id']+'.json'),r);rows.append(r)
            print(r['case_id'],r['status'],r['achieved_impulse_m_s'],r['elapsed_s'],flush=True)
    dump(out/(prefix+'.json'),rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--override');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();run(a.output,workers=a.workers,override=a.override)
