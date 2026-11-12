"""Holdout environment owns events; controllers receive current observations only."""
from pathlib import Path
import json
import time
import concurrent.futures
from collections import Counter
import numpy as np
from scipy.integrate import solve_ivp
from h1_kinematics.assets import dump,sha256
from h1_coupled.contact import CoupledModel
from h1_coupled.validation import validate
from .common import Context,variant
from .control import make_controller


class ScheduledDisturbance:
    def __init__(self,initial_v_m_s=0.,events=()):
        self.initial=float(initial_v_m_s);self.events=[(float(t),float(v)) for t,v in events]
        if any(t<0 or not np.isfinite([t,v]).all() for t,v in self.events):raise ValueError('invalid events')

    def arrays(self,human,n):
        jumps=np.zeros(n+1);jumps[0]=self.initial
        for t,v in self.events:
            k=int(round(t/human.dt_s))
            if abs(k*human.dt_s-t)>1e-10 or k>n:raise ValueError('event not on simulation grid')
            jumps[k]+=v
        return jumps,np.zeros(n)


def handle_state(chart,y,w):
    """Reuse the validated robot FK/J and spatial curve; no new kinematics."""
    q,b=chart.unpack(chart.spline(y));qd=np.zeros(chart.model.robot.model.nv)
    qd[chart.model.active]=chart.spline(y,1)*w
    poses=chart.model.poses(q,b);js=chart.model.jacobians(q,b,False)
    hy=np.array([p[1,3]-p0[1,3] for p,p0 in zip(poses,chart.initial)])
    hv=np.array([(j@qd)[1] for j in js])
    return hy,hv


def actual_step(m,chart,state,a,sample_dt=.005):
    """DOP853 human flow driven by actual FK/J handles, constant joint-path a.

    Reuse the M1-D coupled coefficient matrix, adding only the FK contact
    discrepancy. Robot y/w are exact double integrators on the same chart.
    """
    def handles(t):return handle_state(chart,state[2]+state[3]*t+.5*a*t*t,state[3]+a*t)
    def rhs(t,h):
        y=state[2]+state[3]*t+.5*a*t*t;w=state[3]+a*t;hy,hv=handles(t)
        v=m.M[:2,:2]@h+m.M[:2,2:]@np.array([y,w])
        v[1]+=sum(m.k*(hy-y)+m.c*(hv-w))/m.human.mass_kg
        return v
    ode=solve_ivp(rhs,(0,m.dt),state[:2],method='DOP853',rtol=2e-10,atol=2e-12,dense_output=True)
    if not ode.success:raise ArithmeticError('actual-contact ODE failed: '+ode.message)
    tt=np.linspace(0,m.dt,max(2,int(np.ceil(m.dt/sample_dt)))+1)
    ss=np.c_[ode.sol(tt).T,state[2]+state[3]*tt+.5*a*tt**2,state[3]+a*tt]
    hs=[handles(t) for t in tt];hy=np.array([h[0] for h in hs]);hv=np.array([h[1] for h in hs])
    force=m.k*(hy-ss[:,0,None])+m.c*(hv-ss[:,1,None])
    return tt,ss,hy,hv,force


def inside_box(s,human,feedback):
    s=np.asarray(s)
    return ((abs(s[...,0])<=human.x_terminal_m+1e-9)&(abs(s[...,1])<=human.v_terminal_m_s+1e-9)
        &(abs(s[...,2])<=feedback['terminal_y_m']+1e-9)&(abs(s[...,3])<=feedback['terminal_speed_m_s']+1e-9))


def recovery_metrics(times,states,node_states,terminal_time,total_time,last_event,m,cc,valid):
    """Separate endpoint success from never leaving box in the observation window."""
    times=np.asarray(times);states=np.asarray(states);nodes=np.asarray(node_states)
    complete=len(nodes)==m.human.steps(total_time)+1
    ti=m.human.steps(terminal_time);terminal=bool(len(nodes)>ti and inside_box(nodes[ti],m.human,cc['feedback']))
    flags=inside_box(states,m.human,cc['feedback']);post=times>=terminal_time-1e-9
    sustained=bool(valid and complete and terminal and post.any() and flags[post].all()
        and inside_box(nodes[ti:],m.human,cc['feedback']).all())
    recovery=None
    if valid and complete:
        for i in np.flatnonzero(times>=last_event-1e-9):
            # Require the box to the end, with at least .2 s observed dwell.
            if flags[i:].all() and total_time-times[i]>=cc['feedback']['minimum_settled_s']-1e-9:
                recovery=float(times[i]-last_event);break
    return dict(terminal_recovered=bool(valid and terminal),sustained_recovered=sustained,
        terminal_box_met_before_validity=terminal,observation_complete=complete,recovery_time_s=recovery,
        terminal_time_s=terminal_time,observation_end_s=total_time,
        definition='Endpoint at last event +1.6s; sustained requires all sampled states in box for subsequent 0.8s, same causal controller')


def simulate(m,chart,e,scenario,controller_name,options,cc,terminal_time,total_time):
    disturbance=ScheduledDisturbance(scenario['initial_v_m_s'],scenario['events']);n=m.human.steps(total_time)
    jumps,d=disturbance.arrays(m.human,n);s=np.array([0.,jumps[0],0.,0.]);states=[s.copy()];shadow=[s.copy()]
    control=make_controller(controller_name,m,e,cc['_profile']['delay_s'],options)
    aa=[];logs=[];dense=[];forces=[];hy_all=[];hv_all=[];times=[];failure=None
    for k in range(n):
        started=time.perf_counter();a,log=control.command(s.copy());log['wall_s']=time.perf_counter()-started
        log['missed_control_period']=log['wall_s']>m.dt;logs.append(log)
        if a is None:failure=log.get('reason','controller_no_admissible_action');break
        tt,ss,hy,hv,f=actual_step(m,chart,s,a,cc['experiment']['validation_dt_s'])
        aa.append(a);dense.extend(ss);times.extend(k*m.dt+tt);forces.extend(f);hy_all.extend(hy);hv_all.extend(hv)
        s=ss[-1].copy();s[1]+=jumps[k+1];states.append(s.copy())
        nominal=m.step(shadow[-1],a,d[k]);nominal[1]+=jumps[k+1];shadow.append(nominal)
        if abs(s[0])>m.human.x_safe_m+1e-7 or abs(s[1])>m.human.v_abs_max_m_s+1e-7:
            failure='human_path_domain_exceeded';break
        if s[2]<e['y_lower_m']-1e-7 or s[2]>e['y_upper_m']+1e-7:
            failure='handle_chart_domain_exceeded';break
    return dict(state=np.array(states),shadow=np.array(shadow),acceleration=np.array(aa),command_logs=logs,
        dense_state=np.array(dense).reshape(-1,4),time_s=np.array(times),actual_force=np.array(forces).reshape(-1,2),
        hy=np.array(hy_all).reshape(-1,2),hv=np.array(hv_all).reshape(-1,2),failure=failure,disturbance=disturbance)


CTX=None
def initialize(c,out):
    global CTX
    CTX=Context(c,out)


def compute(case):
    ctx=CTX;c=ctx.c;out=ctx.out;started=time.perf_counter();cc=variant(c,case['variant'])
    m=CoupledModel(cc['_human'],cc['contact'],c['control']['dt_s'])
    scenario=case['scenario'];last=max([t for t,v in scenario['events']]+[0.])
    terminal=last+c['evaluation']['post_last_event_horizon_s'];total=terminal+c['evaluation']['observation_s']
    chart,e,policy,rejected=ctx.controller_chart(case['candidate'],cc)
    r=simulate(m,chart,e,scenario,case['controller'],c['control'],cc,terminal,total)
    path=out/'paths'/(case['case_id']+'.npz')
    # Frozen M1-D validator independently replays the controls against true FK.
    # Its shadow state is the exact ideal contact flow; actual human observations
    # above are saved separately and must agree with this independent ODE replay.
    v=validate(m,chart,e,r['shadow'],r['acceleration'],r['disturbance'],total,cc,path if len(r['acceleration']) else None)
    reasons=list(v['reasons']);ss=r['dense_state'];ff=r['actual_force']
    actual_ode_error=None
    if len(ss):
        with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
        actual_ode_error=float(np.max(abs(data['independent_human_nodes']-r['state'][:,:2])))
        if actual_ode_error>1e-7:reasons.append('actual_closed_loop_replay_disagreement')
        if np.max(abs(ss[:,0]))>m.human.x_safe_m+1e-7:reasons.append('actual_human_position')
        if np.max(abs(ss[:,1]))>m.human.v_abs_max_m_s+1e-7:reasons.append('actual_human_velocity')
        fy_nodes=np.array([m.k*(handle_state(chart,s[2],s[3])[0]-s[0])+m.c*(handle_state(chart,s[2],s[3])[1]-s[1]) for s in r['state']])
        if max(np.max(abs(ff)),np.max(abs(fy_nodes)))>m.cap_y+1e-5:reasons.append('actual_contact_cap')
        # Correct every sampled contact torque using actual human state; all
        # robot quantities are unchanged by human replay.
        budget=ctx.robot.budgets().copy();budget[ctx.robot.torso_indices]=cc['robot']['torso_budget_nm'];mask=np.isfinite(budget)
        tau=[]
        for q,b,f in zip(data['q_rad'],data['base_xy_yaw'],ff):
            tau.append(ctx.robot.contact_map(q,b)@np.r_[0,f[0],m.fz,0,f[1],m.fz])
        tau=np.array(tau);ratio=float(np.max(abs(tau[:,mask])/budget[mask]))
        for s,fy in zip(r['state'],fy_nodes):
            q,b=chart.unpack(chart.spline(s[2]));ntau=ctx.robot.contact_map(q,b)@np.r_[0,fy[0],m.fz,0,fy[1],m.fz]
            ratio=max(ratio,float(np.max(abs(ntau[mask])/budget[mask])))
        if ratio>1+2e-6:reasons.append('actual_contact_torque_budget')
        data.update(actual_state=ss,observed_node_state=r['state'],actual_per_hand_force_n=ff,actual_torque_nm=tau,
            terminal_time_s=np.array(terminal),observation_end_s=np.array(total))
        np.savez_compressed(path,**data)
        peakx=float(np.max(abs(ss[:,0])));peakf=float(max(np.max(abs(ff)),np.max(abs(fy_nodes))))
    else:
        peakx=0.;peakf=abs(m.c*scenario['initial_v_m_s']);ratio=None
        np.savez_compressed(path,observed_node_state=r['state'],acceleration_m_s2=r['acceleration'],
            time_s=np.array([]),actual_state=ss,actual_per_hand_force_n=ff)
    valid=not reasons
    # Include post-jump nodes in box and peak checks; dense traces contain the
    # left limit at event endpoints and the right limit at the next interval.
    tt=np.r_[r['time_s'],np.arange(len(r['state']))*m.dt];st=np.vstack([ss,r['state']]);ix=np.argsort(tt,kind='stable')
    rm=recovery_metrics(tt[ix],st[ix],r['state'],terminal,total,last,m,cc,valid)
    durations=[l['wall_s'] for l in r['command_logs']]
    report=dict(case,**rm,valid=valid,failures=sorted(set(reasons+([r['failure']] if r['failure'] else []))),
        completed_intervals=len(r['acceleration']),requested_intervals=m.human.steps(total),
        path_file=str(path),chart_policy=policy,envelope=e,rejected_envelopes=rejected,
        peak_human_x_m=max(peakx,float(np.max(abs(r['state'][:,0])))),peak_per_hand_force_n=peakf,
        actual_max_torque_ratio=ratio,actual_vs_replayed_ode_error=actual_ode_error,validation=v,
        command_logs=r['command_logs'],controller_calls=len(durations),
        fallback_calls=sum(l.get('fallback',False) for l in r['command_logs']),
        qp_status_counts=dict(Counter(l.get('qp_status','not_applicable') for l in r['command_logs'])),
        control_wall_mean_s=float(np.mean(durations)),control_wall_max_s=max(durations),
        missed_control_periods=sum(t>m.dt for t in durations),elapsed_s=time.perf_counter()-started)
    return report


def cases(c):
    scenarios=[('nominal',s) for s in c['test_scenarios']]
    for name in c['evaluation']['variants']:
        if name=='nominal':continue
        for dv in c['evaluation']['sensitivity_impulses_m_s']:
            scenarios.append((name,dict(id=f'pulse_{dv:+.2f}',initial_v_m_s=0.,events=[[c['evaluation']['sensitivity_onset_s'],dv]])))
    result=[];rng=np.random.default_rng(c['seed'])
    for name,s in scenarios:
        random_id=str(rng.choice(c['candidates'])) # fixed before any outcome is read
        for candidate in c['candidates']:
            for control in ['fixed','mpc']:
                result.append(dict(case_id=f"{name}_{s['id']}_{candidate}_{control}",variant=name,scenario=s,
                    candidate=candidate,controller=control,random_draw_candidate=random_id))
    return result


def run(c,out,workers=4):
    out=Path(out);selection=sha256(out/'selection.json');freeze=out/'test_selection_freeze.json'
    if freeze.exists():
        if json.loads(freeze.read_text())['sha256']!=selection:raise ValueError('selection changed since original holdout start')
    else:dump(freeze,dict(sha256=selection))
    for name in ['cases','paths']:(out/name).mkdir(exist_ok=True)
    plan=cases(c);dump(out/'test_plan.json',plan)
    pending=[x for x in plan if not (out/'cases'/(x['case_id']+'.json')).exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(c,out)) as pool:
        for i,r in enumerate(pool.map(compute,pending,chunksize=1)):
            dump(out/'cases'/(r['case_id']+'.json'),r)
            if i%16==0:print('test',i+1,'/',len(pending),r['case_id'],r['terminal_recovered'],r['sustained_recovered'],flush=True)
    if sha256(out/'selection.json')!=selection:raise ValueError('selection changed during holdout')
    rows=[json.loads((out/'cases'/(x['case_id']+'.json')).read_text()) for x in plan]
    dump(out/'closed_loop.json',rows)
    print('terminal',sum(r['terminal_recovered'] for r in rows),'sustained',sum(r['sustained_recovered'] for r in rows),'/',len(rows),flush=True)
