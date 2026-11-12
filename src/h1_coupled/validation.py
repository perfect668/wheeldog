"""Actual FK/J/contact replay, independent ODE, sampled supplied-mesh checks."""
import numpy as np
from scipy.integrate import solve_ivp
from h1_capability.reachability import pose_error


def validate(model,chart,envelope,state,acceleration,disturbance,horizon,config,save_path=None):
    n=len(acceleration);dt=model.dt;jumps,d=disturbance.arrays(model.human,model.human.steps(horizon))
    profile=chart.profile;robot=chart.model;budget=robot.budgets().copy()
    budget[robot.torso_indices]=config['robot']['torso_budget_nm']
    mask=np.isfinite(budget);sub=max(2,int(np.ceil(dt/config['experiment']['validation_dt_s'])))
    records=[];reasons=[];pe=re=0.;clearance=np.inf;ode_states=[np.array([0.,jumps[0]])]
    peakqv=peakqa=peakbasev=peakbasea=torque_ratio=force_ratio=0.;force_error=0.
    def robot_at(y,w,a):
        z=chart.spline(y);zd=chart.spline(y,1)*w;zdd=chart.spline(y,2)*w*w+chart.spline(y,1)*a
        q,b=chart.unpack(z);poses=robot.poses(q,b)
        qdot=np.zeros(robot.robot.model.nv);qdot[robot.active]=zd[:chart.n]
        js=robot.jacobians(q,b,chart.mobile)
        velocity=[j@np.r_[qdot,zd[-3:]] if chart.mobile else j@qdot for j in js]
        handle_y=np.array([p[1,3]-p0[1,3] for p,p0 in zip(poses,chart.initial)])
        handle_v=np.array([v[1] for v in velocity])
        return q,b,poses,velocity,handle_y,handle_v,zd,zdd
    for k,a in enumerate(acceleration):
        for si,t in enumerate(np.linspace(0,dt,sub+1)):
            s=model.step(state[k],a,d[k],float(t));x,v,y,w=s
            q,b,poses,velocity,hy,hv,zd,zdd=robot_at(y,w,a)
            actual_force=model.k*(hy-x)+model.c*(hv-v)
            targets=[p.copy() for p in chart.initial]
            for p in targets:p[1,3]+=y
            error=pose_error(poses,targets).reshape(2,6)
            pe=max(pe,float(max(np.linalg.norm(error[:,:3],axis=1))));re=max(re,float(max(np.linalg.norm(error[:,3:],axis=1))))
            force_error=max(force_error,float(max(abs(actual_force-model.per_hand_force(s)))))
            wrench=np.r_[0.,actual_force[0],model.fz,0,actual_force[1],model.fz];tau=robot.contact_map(q,b)@wrench
            torque_ratio=max(torque_ratio,float(max(abs(tau[mask])/budget[mask])))
            force_ratio=max(force_ratio,float(max(np.hypot(actual_force,model.fz))/model.contact['resultant_cap_per_hand_n']))
            peakqv=max(peakqv,float(max(abs(zd[:chart.n])/robot.velocity)))
            peakqa=max(peakqa,float(max(abs(zdd[:chart.n])/robot.coordinate_acceleration(profile,False))))
            if chart.mobile:
                peakbasev=max(peakbasev,float(np.linalg.norm(zd[-3:-1])/robot.config['base_velocity_m_s']),abs(zd[-1])/profile['base_yaw_velocity_rad_s'])
                peakbasea=max(peakbasea,float(np.linalg.norm(zdd[-3:-1])/profile['base_acceleration_m_s2']),abs(zdd[-1])/profile['base_yaw_acceleration_rad_s2'])
            if np.any(q<robot.robot.lower-1e-8) or np.any(q>robot.robot.upper+1e-8):reasons.append('joint_position')
            if y<envelope['y_lower_m']-1e-8 or y>envelope['y_upper_m']+1e-8:reasons.append('chart_domain')
            if si%config['experiment']['collision_stride']==0 or si==sub:
                collision=chart.checker.check(robot.robot.fk(q,b),config['robot']['clearance_m'])
                clearance=min(clearance,collision['min_distance_m'])
                if not collision['passed']:reasons.append('sampled_collision')
            records.append((k*dt+t,s,q,b,actual_force,hy,hv,tau,zd,zdd))
        # Independent high-accuracy integration, actual nonlinear handles driving
        # the original M0 LIPM RHS, with no assumed force interpolation.
        def rhs(t,human):
            y=state[k,2]+state[k,3]*t+.5*a*t*t;w=state[k,3]+a*t
            _,_,_,_,hy,hv,_,_=robot_at(y,w,a)
            f=model.k*(hy-human[0])+model.c*(hv-human[1])
            return [human[1],model.human.omega**2*human[0]+(sum(f)+d[k])/model.human.mass_kg]
        ode=solve_ivp(rhs,(0,dt),ode_states[-1],rtol=2e-10,atol=2e-12,method='DOP853')
        if not ode.success:reasons.append('independent_ode_failure')
        end=ode.y[:,-1];end[1]+=jumps[k+1];ode_states.append(end)
    if not records:
        return dict(valid=False,terminal_met=False,reasons=['no_integrated_intervals'])
    times=np.array([r[0] for r in records]);ss=np.array([r[1] for r in records]);contact=np.array([r[4] for r in records])
    ode_states=np.array(ode_states);ode_error=np.max(abs(ode_states-state[:,:2]),axis=0)
    # Include post-jump endpoints, which can have immediate damping-force changes.
    postforce=model.per_hand_force(state)
    if max(abs(postforce))>model.cap_y+1e-5:reasons.append('post_jump_contact_cap')
    if np.max(abs(state[:,0]))>model.human.x_safe_m+1e-7 or max(abs(ss[:,0]))>model.human.x_safe_m+1e-7:reasons.append('human_position')
    if np.max(abs(state[:,1]))>model.human.v_abs_max_m_s+1e-7 or max(abs(ss[:,1]))>model.human.v_abs_max_m_s+1e-7:reasons.append('human_velocity')
    if pe>config['robot']['position_tolerance_m'] or re>config['robot']['rotation_tolerance_rad']:reasons.append('handle_coordination')
    if max(peakqv,peakqa,peakbasev,peakbasea)>1+2e-6:reasons.append('robot_velocity_or_acceleration')
    if torque_ratio>1+2e-6:reasons.append('contact_torque_budget')
    if force_ratio>1+2e-6:reasons.append('contact_resultant_cap')
    if max(ode_error)>2e-5:reasons.append('nonlinear_contact_ode_disagreement')
    dynamics=0.
    for k,a in enumerate(acceleration):
        prediction=model.step(state[k],a,d[k]);prediction[1]+=jumps[k+1]
        dynamics=max(dynamics,float(max(abs(prediction-state[k+1]))))
    if dynamics>1e-7:reasons.append('coupled_dynamics')
    complete=n==model.human.steps(horizon)
    terminal=bool(complete and abs(ode_states[-1,0])<=model.human.x_terminal_m+1e-7
        and abs(ode_states[-1,1])<=model.human.v_terminal_m_s+1e-7
        and abs(state[-1,2])<=config['feedback']['terminal_y_m'] and abs(state[-1,3])<=config['feedback']['terminal_speed_m_s'])
    inside=(abs(ode_states[:,0])<=model.human.x_terminal_m)&(abs(ode_states[:,1])<=model.human.v_terminal_m_s)&(abs(state[:,2])<=config['feedback']['terminal_y_m'])&(abs(state[:,3])<=config['feedback']['terminal_speed_m_s'])
    last_disturbance=max(np.flatnonzero(abs(jumps[:n+1])>0).tolist()+np.flatnonzero(abs(d[:n])>0).tolist()+[0])
    recovery=None
    for k in range(last_disturbance,n+1):
        if inside[k:].all() and (n-k)*dt>=config['feedback']['minimum_settled_s']-1e-9:recovery=k*dt;break
    result=dict(valid=not reasons,terminal_met=terminal,complete=complete,reasons=sorted(set(reasons)),
        recovery_time_s=recovery,dynamics_max_abs_error=dynamics,
        independent_ode_x_error_m=float(ode_error[0]),independent_ode_v_error_m_s=float(ode_error[1]),
        max_handle_position_error_m=pe,max_handle_rotation_error_rad=re,contact_force_model_max_error_n=force_error,
        max_joint_velocity_ratio=peakqv,max_joint_acceleration_ratio=peakqa,max_base_velocity_ratio=peakbasev,max_base_acceleration_ratio=peakbasea,
        max_torque_ratio=torque_ratio,max_contact_cap_ratio=force_ratio,min_sampled_clearance_m=clearance,
        peak_human_x_m=float(max(max(abs(ss[:,0])),max(abs(state[:,0])))),
        peak_human_v_m_s=float(max(max(abs(ss[:,1])),max(abs(state[:,1])))),
        peak_per_hand_lateral_force_n=float(max(np.max(abs(contact)),max(abs(postforce)))),
        peak_handle_displacement_m=float(max(abs(ss[:,2]))),peak_handle_speed_m_s=float(max(abs(ss[:,3]))),
        terminal_state=state[-1].tolist(),terminal_independent_human=ode_states[-1].tolist(),
        interpretation='Sampled geometry/contact constraints and independent human ODE; no continuous mesh guarantee or invariant terminal claim')
    if save_path is not None:
        np.savez_compressed(save_path,time_s=times,state=ss,node_state=state,node_time_s=np.arange(n+1)*dt,
            acceleration_m_s2=acceleration,independent_human_nodes=ode_states,
            q_rad=np.array([r[2] for r in records]),base_xy_yaw=np.array([r[3] for r in records]),
            per_hand_lateral_force_n=contact,handle_y_m=np.array([r[5] for r in records]),handle_v_m_s=np.array([r[6] for r in records]),
            torque_nm=np.array([r[7] for r in records]),coordinate_velocity=np.array([r[8] for r in records]),
            coordinate_acceleration=np.array([r[9] for r in records]),jumps_m_s=jumps,disturbance_force_n=d,
            envelope_json=np.array(__import__('json').dumps(envelope)))
    return result
