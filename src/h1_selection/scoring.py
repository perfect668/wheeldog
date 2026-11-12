"""Offline directional recovery lower bounds using M1-D direct optimization."""
from pathlib import Path
import copy
import json
import time
import concurrent.futures
import hashlib
import numpy as np
from h1_kinematics.assets import dump
from h1_coupled.contact import CoupledModel
from h1_coupled.direct import optimize_iteration,replay
from h1_coupled.solver import solve_offline
from recoverability.disturbance import Disturbance
from h1_capability.force import ray_capacity
from h1_capability.instantaneous import cooperative_velocity
from .common import Context,variant


CONTEXT=None
def init(c,out):
    global CONTEXT
    CONTEXT=Context(c,out)


def score_case(arguments):
    candidate,sign=arguments;ctx=CONTEXT;c=ctx.c;out=ctx.out;started=time.perf_counter()
    cc=variant(c,'nominal');cc['direct'].update(c['score']);m=CoupledModel(cc['_human'],cc['contact'],c['control']['dt_s'])
    h=c['score']['horizon_s'];n=m.human.steps(h);nj=len(ctx.robot.active)
    chart,_,_=ctx.chart(candidate,'speed_weighted',cc);attempts=[];best=None;seeds=[]
    for label in c['score']['starts']:
        ref=np.zeros((n+1,nj));seedlog=dict(start=label,status='ZERO')
        if label!='zero':
            curve,env,_=ctx.chart(candidate,label,cc)
            seedlog['status']='NO_FEASIBLE_CHART_SEED'
            # Same +/-0.08m/s scoring seed for all, try all common domains.
            for e in env:
                r=solve_offline(m,Disturbance(delta_v_m_s=sign*c['score']['seed_impulse_m_s']),h,e,cc['_profile']['delay_s'])
                if r['status']=='LP_FEASIBLE':
                    ref=curve.spline(r['state'][:,2])[:,:nj]-curve.q0[ctx.robot.active]
                    seedlog.update(status='LINEAR_CHART_SEED',envelope=e);break
        seedlog['reference_sha256']=hashlib.sha256(ref.tobytes()).hexdigest()
        seedlog['duplicate_of']=next((s['start'] for s in seeds if s['reference_sha256']==seedlog['reference_sha256']),None)
        seeds.append(seedlog)
        for i in range(c['score']['sequential_iterations']):
            try:r=optimize_iteration(chart,m,ref,h,sign,cc,trust=c['score']['trust_region_rad'])
            except (ValueError,FloatingPointError,RuntimeError) as exc:
                attempts.append(dict(start=label,iteration=i,status='NUMERICAL_EXCEPTION',message=f'{type(exc).__name__}: {exc}'));break
            log={k:v for k,v in r.items() if k not in ['states','acceleration']};log.update(start=label,iteration=i)
            if r['status']!='LINEARIZED_OPTIMUM':attempts.append(log);break
            raw=out/'score_iterates'/f'{candidate}_{sign:+d}_{label}_{i:02d}.npz'
            np.savez_compressed(raw,states=r['states'],acceleration=r['acceleration'],impulse_m_s=r['impulse_m_s'],reference=ref)
            log['iterate_file']=str(raw)
            v=replay(chart,m,r['states'],r['acceleration'],r['impulse_m_s'],sign,cc)
            log['validation']=v;attempts.append(log)
            if v['valid'] and (best is None or r['impulse_m_s']>best['impulse_m_s']):best=dict(r,metrics=v,start=label,iteration=i)
            ref=r['states'][:,2:2+nj].copy()
    path=out/'score_paths'/f'{candidate}_{sign:+d}.npz'
    if best:replay(chart,m,best['states'],best['acceleration'],best['impulse_m_s'],sign,cc,path)
    bystart={label:max([a['impulse_m_s'] for a in attempts if a['start']==label and a.get('validation',{}).get('valid')]+[0.]) for label in c['score']['starts']}
    return dict(candidate=candidate,sign=sign,lower_bound_m_s=best['impulse_m_s'] if best else 0.,
        status='VERIFIED_LOWER_BOUND' if best else 'NO_NONZERO_VERIFIED_LOWER_BOUND',
        path_file=str(path) if best else None,metrics=best['metrics'] if best else None,best_start=best['start'] if best else None,
        seeds=seeds,bounds_by_start=bystart,attempts=attempts,elapsed_s=time.perf_counter()-started,
        information='Offline amplitude maximization, same scenario class and search budget; NOT global maximum or online controller')


def run(c,out,workers=4):
    out=Path(out)
    for name in ['score_cases','score_iterates','score_paths']:(out/name).mkdir(exist_ok=True)
    arguments=[(id,sign) for id in c['candidates'] for sign in c['score']['signs']]
    pending=[a for a in arguments if not (out/'score_cases'/f'{a[0]}_{a[1]:+d}.json').exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(c,out)) as pool:
        for r in pool.map(score_case,pending,chunksize=1):
            dump(out/'score_cases'/f"{r['candidate']}_{r['sign']:+d}.json",r)
            print('score',r['candidate'],r['sign'],r['status'],r['lower_bound_m_s'],r['elapsed_s'],flush=True)
    rows=[json.loads((out/'score_cases'/f'{id}_{sign:+d}.json').read_text()) for id,sign in arguments]
    dump(out/'directional_scores.json',rows)


def select(c,out):
    """No test inputs are read. Freeze all four policies before online tests."""
    out=Path(out);ctx=Context(c,out);cc=variant(c,'nominal');scores=json.loads((out/'directional_scores.json').read_text());rows=[]
    for id in c['candidates']:
        chart,_,_=ctx.chart(id,'speed_weighted',cc);robot=ctx.robot;q=chart.q0;b=chart.b0
        budget=robot.budgets().copy();budget[robot.torso_indices]=cc['robot']['torso_budget_nm']
        task=np.tile([0.,0.,cc['contact']['vertical_per_hand_n']],2);A=robot.contact_map(q,b)
        force=[ray_capacity(A,budget,[0,sign,0],task) for sign in [-1,1]]
        speed=[cooperative_velocity(robot,q,b,np.array([0,sign,0]),cc['_profile'],False) for sign in [-1,1]]
        if any(x['status']!='OPTIMAL' for x in force+speed):raise ValueError('static metric failure: '+id)
        directional=[next(s['lower_bound_m_s'] for s in scores if (s['candidate'],s['sign'])==(id,sign)) for sign in [-1,1]]
        tau=A@task;mask=np.isfinite(budget)
        rows.append(dict(candidate=id,R_negative_m_s=directional[0],R_positive_m_s=directional[1],
            recovery_min_m_s=min(directional),recovery_mean_m_s=float(np.mean(directional)),
            force_proxy_min_n=min(f['capacity_n'] for f in force),
            contact_capped_force_min_n=min(min(f['capacity_n'] for f in force),2*np.sqrt(cc['contact']['resultant_cap_per_hand_n']**2-cc['contact']['vertical_per_hand_n']**2)),
            instantaneous_min_m_s=min(s['speed_m_s'] for s in speed),static_force=force,instantaneous=speed,
            task_max_torque_ratio=float(max(abs(tau[mask])/budget[mask])),
            current_per_hand_force_n=[0.,0.,cc['contact']['vertical_per_hand_n']],q_rad=q.tolist(),base_xy_yaw=b.tolist(),
            world_handle_poses=np.array(chart.initial).tolist()))
    def choose(key):return max(rows,key=lambda r:(r[key],-c['candidates'].index(r['candidate'])))['candidate']
    methods=dict(A_default=c['selection']['default'],B_force=choose('force_proxy_min_n'),C_kinematic=choose('instantaneous_min_m_s'),D_future=choose('recovery_min_m_s'))
    selected=dict(methods=methods,random_uniform_candidates=c['candidates'],metrics=rows,
        core_four_methods=dict(A_default=c['selection']['default'],**{name:max(rows[:4],key=lambda r:(r[key],-c['candidates'].index(r['candidate'])))['candidate'] for name,key in [('B_force','force_proxy_min_n'),('C_kinematic','instantaneous_min_m_s'),('D_future','recovery_min_m_s')]}),
        source='Only initial geometry, nominal model and offline scoring set; no test outcome read',
        force_caveat='B ranks nominal contact-torque-only residual rays; capped actual contact metric may tie. No kN ray is delivered to human.')
    dump(out/'selection.json',selected)
    print(json.dumps(dict(methods=methods,metrics=[{k:v for k,v in r.items() if k in ['candidate','recovery_min_m_s','recovery_mean_m_s','force_proxy_min_n','contact_capped_force_min_n','instantaneous_min_m_s']} for r in rows]),indent=2))
