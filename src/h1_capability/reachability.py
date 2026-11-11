"""Offline cooperative path continuation, not MPC or a complete reachability oracle.

A common minimum-jerk translation is tracked by local weighted pseudoinverse
continuation. Both handle orientations are fixed. Piecewise cubic Hermite joint
and base paths start/stop at rest, respect an explicit delay, and are checked
for continuous scalar kinematic bounds plus sampled collision/pose constraints.
The best accepted path is a lower bound in this declared path family.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicHermiteSpline


def pose_error(poses,targets):
    return np.concatenate([np.r_[p[:3,3]-t[:3,3],Rotation.from_matrix(p[:3,:3]@t[:3,:3].T).as_rotvec()]
                           for p,t in zip(poses,targets)])


def polynomial_roots_in_unit(coeff):
    coeff=np.trim_zeros(np.asarray(coeff,float),'f')
    if len(coeff)<2:return []
    return [float(x.real) for x in np.roots(coeff) if abs(x.imag)<1e-9 and 0<x.real<1]


def spline_extrema(spline,times):
    """Exact scalar q/v/a maxima of cubic segments and xy vector v/a norms."""
    c=spline.c;n=c.shape[2]
    qmin=np.full(n,np.inf);qmax=-qmin.copy();vmax=np.zeros(n);amax=np.zeros(n)
    xy_speed=0.;xy_accel=0.
    for i,h in enumerate(np.diff(times)):
        # Polynomial in u=(t-t_i)/h, coefficients descending.
        p=c[:,i]*np.array([h**3,h**2,h,1.])[:,None]
        for k in range(n):
            a,b,cc,d=p[:,k]
            pos=[0.,1.]+polynomial_roots_in_unit([3*a,2*b,cc])
            vel=[0.,1.]+polynomial_roots_in_unit([6*a,2*b])
            values=np.polyval(p[:,k],pos)
            qmin[k]=min(qmin[k],float(min(values)));qmax[k]=max(qmax[k],float(max(values)))
            vmax[k]=max(vmax[k],float(max(abs(np.polyval([3*a,2*b,cc],vel)))/h))
            amax[k]=max(amax[k],float(max(abs(np.array([2*b,6*a+2*b])))/h**2))
        if n>=3:
            # Caller includes xy only in final 3 coordinates for mobile case;
            # compute even in fixed case but ignore these fields there.
            vx=np.array([3*p[0,-3],2*p[1,-3],p[2,-3]])/h
            vy=np.array([3*p[0,-2],2*p[1,-2],p[2,-2]])/h
            square=np.polyadd(np.polymul(vx,vx),np.polymul(vy,vy))
            loc=[0,1]+polynomial_roots_in_unit(np.polyder(square))
            xy_speed=max(xy_speed,float(np.sqrt(max(0,max(np.polyval(square,loc))))))
            ax=np.polyder(vx)/h;ay=np.polyder(vy)/h
            xy_accel=max(xy_accel,float(max(np.hypot(np.polyval(ax,[0,1]),np.polyval(ay,[0,1])))))
    return dict(q_min=qmin,q_max=qmax,v_abs_max=vmax,a_abs_max=amax,
                base_speed_norm_max=xy_speed,base_accel_norm_max=xy_accel)


def construct_path(model,q0,base0,direction,displacement,horizon,profile,mobile,options,policy):
    delay=profile['delay_s'];dt=options['dt_s'];n=len(model.active)+(3 if mobile else 0)
    if horizon<=delay:
        return dict(status='KINEMATIC_REJECTED',reasons=['delay_exhausts_window'])
    times=np.unique(np.r_[np.arange(0,horizon,dt),horizon,delay])
    u=np.clip((times-delay)/(horizon-delay),0,1)
    progress=displacement*(10*u**3-15*u**4+6*u**5)
    initial=model.poses(q0,base0);q=q0.copy();base=base0.copy();coordinates=[]
    vmax=model.coordinate_velocity(profile,mobile)
    weights=np.ones(n) if policy=='uniform' else vmax**2
    for t,s in zip(times,progress):
        targets=[p.copy() for p in initial]
        for p in targets:p[:3,3]+=s*direction
        if t<=delay:q=q0.copy();base=base0.copy()
        else:
            converged=False
            for _ in range(options['projection_max_iterations']):
                error=pose_error(model.poses(q,base),targets)
                if max(abs(error))<=options['projection_tolerance']:converged=True;break
                j=model.task_jacobian(q,base,mobile)
                # Weighted minimum norm correction. No old-module rewrite.
                weighted=j*np.sqrt(weights)[None,:]
                delta=np.sqrt(weights)*np.linalg.lstsq(weighted,-error,rcond=1e-9)[0]
                if not np.isfinite(delta).all() or np.max(abs(delta))>.8:
                    return dict(status='KINEMATIC_REJECTED',reasons=['local_projection_step'])
                q,base=model.update(q,base,delta,mobile)
            if not converged and max(abs(pose_error(model.poses(q,base),targets)))>options['projection_tolerance']:
                return dict(status='KINEMATIC_REJECTED',reasons=['local_projection_residual'])
        coordinates.append(np.r_[q[model.active],base] if mobile else q[model.active].copy())
    coordinates=np.array(coordinates)
    slopes=np.zeros_like(coordinates)
    # Tangents satisfy the common Cartesian velocity at every knot, rather than
    # central chord slopes that needlessly introduce interpolation pose drift.
    for i,(t,x) in enumerate(zip(times,coordinates)):
        if t<=delay or i==len(times)-1:continue
        qi=q0.copy();qi[model.active]=x[:len(model.active)];bi=x[-3:] if mobile else base0
        j=model.task_jacobian(qi,bi,mobile)
        ds=displacement*(30*u[i]**2-60*u[i]**3+30*u[i]**4)/(horizon-delay)
        twist=np.tile(np.r_[ds*direction,[0,0,0]],2)
        slopes[i]=np.sqrt(weights)*np.linalg.lstsq(j*np.sqrt(weights)[None,:],twist,rcond=1e-9)[0]
    spline=CubicHermiteSpline(times,coordinates,slopes)
    extremes=spline_extrema(spline,times)
    reasons=[];limits=[]
    lo=model.robot.lower[model.active];hi=model.robot.upper[model.active]
    count=len(model.active)
    if np.any(extremes['q_min'][:count]<lo-1e-9) or np.any(extremes['q_max'][:count]>hi+1e-9):reasons.append('joint_position')
    vv=extremes['v_abs_max']/vmax; aa=extremes['a_abs_max']/model.coordinate_acceleration(profile,mobile)
    if np.max(vv)>1+1e-8:reasons.append('joint_or_yaw_velocity')
    if np.max(aa)>1+1e-8:reasons.append('joint_or_yaw_acceleration')
    if mobile:
        if extremes['base_speed_norm_max']>model.config['base_velocity_m_s']+1e-8:reasons.append('base_translation_velocity')
        if extremes['base_accel_norm_max']>profile['base_acceleration_m_s2']+1e-8:reasons.append('base_translation_acceleration')
    names=[model.robot.names[i] for i in model.active]+(['base_x','base_y','base_yaw'] if mobile else [])
    info=dict(peak_velocity_ratio=float(max(vv)),peak_acceleration_ratio=float(max(aa)),
              max_velocity_coordinate=names[int(np.argmax(vv))],max_acceleration_coordinate=names[int(np.argmax(aa))],
              base_speed_norm_max_m_s=extremes['base_speed_norm_max'] if mobile else 0.,
              base_accel_norm_max_m_s2=extremes['base_accel_norm_max'] if mobile else 0.)
    return dict(status='KINEMATIC_REJECTED' if reasons else 'NEEDS_COLLISION_CHECK',reasons=reasons,
        policy=policy,displacement_m=float(displacement),times=times,coordinates=coordinates,slopes=slopes,
        spline=spline,initial_poses=initial,extremes=extremes,**info)


def validate_geometry(model,checker,path,q0,base0,direction,horizon,profile,mobile,options,task):
    times=path['times'];spline=path['spline'];checks=[];position_error=0.;rotation_error=0.;minimum=1e10
    # Includes knots + all uniform interior fractions. Collision is sampled,
    # not certified continuous-time collision detection.
    subs=options['collision_substeps']
    samples=sorted(set(float(t) for a,b in zip(times[:-1],times[1:]) for t in np.linspace(a,b,subs+1)))
    a_map=None;budgets=model.budgets();mask=np.isfinite(budgets)
    for t in samples:
        x=spline(t);q=q0.copy();q[model.active]=x[:len(model.active)];base=x[-3:] if mobile else base0
        u=np.clip((t-profile['delay_s'])/(horizon-profile['delay_s']),0,1)
        progress=path['displacement_m']*(10*u**3-15*u**4+6*u**5)
        targets=[p.copy() for p in path['initial_poses']]
        for p in targets:p[:3,3]+=progress*direction
        error=pose_error(model.poses(q,base),targets).reshape(2,6)
        pe=float(max(np.linalg.norm(error[:,:3],axis=1)));re=float(max(np.linalg.norm(error[:,3:],axis=1)))
        position_error=max(position_error,pe);rotation_error=max(rotation_error,re)
        if pe>options['pose_position_tolerance_m'] or re>options['pose_rotation_tolerance_rad']:
            return dict(status='COORDINATION_REJECTED',reasons=['interpolated_handle_pose'],failed_time_s=t,
                        position_error_m=pe,rotation_error_rad=re)
        tau=model.contact_map(q,base)@task
        if np.max(abs(tau[mask])-budgets[mask])>1e-6:
            return dict(status='TASK_TORQUE_REJECTED',reasons=['task_force_budget_during_motion'],failed_time_s=t)
        collision=checker.check(model.robot.fk(q,base),options['self_clearance_m'])
        minimum=min(minimum,collision['min_distance_m'])
        if not collision['passed']:
            return dict(status='COLLISION_REJECTED',reasons=['sampled_mesh_collision_or_clearance'],failed_time_s=t,collision=collision)
        checks.append(t)
    return dict(status='PATH_ACCEPTED',reasons=[],sample_count=len(checks),max_position_error_m=position_error,
                max_rotation_error_rad=rotation_error,min_sampled_mesh_distance_m=minimum)


def serialize_attempt(result):
    exclude={'times','coordinates','slopes','spline','initial_poses','extremes'}
    return {k:v for k,v in result.items() if k not in exclude}


def search(model,checker,q0,base0,direction,horizon,profile,mobile,options,task,save_path=None):
    started=time.perf_counter();attempts=[];best=None;kinematic_paths=[]
    def test(distance):
        nonlocal best
        feasible=False
        for policy in options['inverse_policies']:
            try:
                result=construct_path(model,q0,base0,direction,distance,horizon,profile,mobile,options,policy)
                attempts.append(dict(distance_m=float(distance),**serialize_attempt(result)))
                if result['status']=='NEEDS_COLLISION_CHECK':
                    feasible=True
                    kinematic_paths.append(result)
                    break # same deterministic policy ordering for every configuration
            except (ValueError,RuntimeError,np.linalg.LinAlgError,FloatingPointError) as exc:
                attempts.append(dict(policy=policy,distance_m=float(distance),status='NUMERICAL_ERROR',reasons=[type(exc).__name__],error=str(exc)))
        return feasible
    cap=options['max_displacement_m']
    # Check a complete coarse grid first: do not assume global monotonicity of IK/collision feasibility.
    grid=np.linspace(0,cap,5);success=[test(float(x)) for x in grid]
    lo=max((x for x,ok in zip(grid,success) if ok),default=0.)
    upper=next((x for x in grid if x>lo),None)
    if upper is not None:
        for _ in range(options['bisection_iterations']):
            mid=(lo+upper)/2
            if test(float(mid)):lo=mid
            else:upper=mid
    # Collision/pose checks on descending candidate paths, after cheap analytic
    # kinematic screening. Every accepted output still checks the entire path.
    def geometric_test(distance,cached=None):
        nonlocal best
        for policy in options['inverse_policies']:
            candidate=cached if cached is not None and cached['policy']==policy else construct_path(
                model,q0,base0,direction,distance,horizon,profile,mobile,options,policy)
            if candidate['status']!='NEEDS_COLLISION_CHECK':
                attempts.append(dict(distance_m=float(distance),**serialize_attempt(candidate)));continue
            check=validate_geometry(model,checker,candidate,q0,base0,direction,horizon,profile,mobile,options,task)
            attempts.append(dict(distance_m=float(distance),policy=policy,**check))
            if check['status']=='PATH_ACCEPTED':
                if best is None or distance>best['displacement_m']:
                    best=candidate.copy();best.update(check)
                return True
        return False
    if kinematic_paths:
        largest=max(kinematic_paths,key=lambda p:p['displacement_m'])
        if not geometric_test(largest['displacement_m'],largest):
            safe=0.;unsafe=largest['displacement_m'];geometric_test(0.)
            for _ in range(options['bisection_iterations']):
                mid=(safe+unsafe)/2
                if geometric_test(mid):safe=mid
                else:unsafe=mid
    if best is not None and save_path is not None:
        np.savez_compressed(save_path,times_s=best['times'],coordinates=best['coordinates'],slopes=best['slopes'],
            q0_rad=q0,base0_xy_yaw=base0,active_indices=np.array(model.active),mobile=np.array(mobile),
            direction_world=direction,horizon_s=np.array(horizon),delay_s=np.array(profile['delay_s']),
            displacement_m=np.array(best['displacement_m']),initial_handle_poses=np.array(best['initial_poses']))
    return dict(status='VERIFIED_PATH_FOUND' if best is not None else 'NO_VERIFIED_PATH',
        achieved_displacement_m=best['displacement_m'] if best else None,
        capped=bool(best is not None and best['displacement_m']>=cap-1e-10),
        local_search_upper_m=float(upper) if upper is not None else None,
        resolution_m=cap/4/2**options['bisection_iterations'],
        best=None if best is None else serialize_attempt(best),attempts=attempts,
        elapsed_s=time.perf_counter()-started,
        interpretation='verified sampled path-family lower bound; local rejection is not global infeasibility')
