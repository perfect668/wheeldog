"""Reproducible M1-D stages; artifacts and source provenance remain separate."""
from pathlib import Path
import argparse
import copy
import json
import time
import subprocess
import concurrent.futures
from collections import Counter
import numpy as np
import yaml
from scipy.interpolate import CubicHermiteSpline
from h1_kinematics.assets import dump,sha256
from h1_capability.experiment import setup as capability_setup,metadata_record
from recoverability.disturbance import Disturbance
from .chart import SpatialChart
from .contact import CoupledModel
from .solver import solve_offline,simulate_feedback
from .validation import validate


def load_config(path):
    c=yaml.safe_load(Path(path).read_text());cap=yaml.safe_load(Path(c['capability_config']).read_text())
    c['_capability']=cap;c['_human']=yaml.safe_load(Path(c['human_config']).read_text())['model']
    c['_profile']=copy.deepcopy(cap['profiles'][c['robot']['profile']])
    return c


def variant_config(c,name):
    cc=copy.deepcopy(c)
    if name=='main':return cc
    for key,value in c['sensitivity']['variants'][name].items():
        if key in cc['contact']:cc['contact'][key]=value
        elif key in cc['robot']:cc['robot'][key]=value
        elif key in cc['_profile']:cc['_profile'][key]=value
        elif key=='dt_s':cc['experiment'][key]=value
        else:raise ValueError('unknown variant parameter '+key)
    return cc


def check_frozen():
    checked={}
    for inventory in ['results/diagnostics/artifact_sha256.json','results/m1/artifact_sha256.json','results/m1c/artifact_sha256.json']:
        hashes=json.loads(Path(inventory).read_text())
        for name,h in hashes.items():
            if sha256(name)!=h:raise ValueError('frozen artifact changed: '+name)
        checked[inventory]=dict(count=len(hashes),all_match=True,inventory_sha256=sha256(inventory))
    return checked


def prepare(c,out):
    out.mkdir(parents=True,exist_ok=True);(out/'charts').mkdir(exist_ok=True)
    saved=out/'config.yaml'
    if saved.exists():raise ValueError('prepare requires a new output directory')
    saved.write_text(yaml.safe_dump(c,sort_keys=False))
    metadata=metadata_record(c['_capability']);metadata['frozen_integrity']=check_frozen()
    metadata['capability_seed']=metadata['seed'];metadata['seed']=c['seed']
    metadata['gravity_model']='Robot contact-only budgets exclude robot gravity; human uses explicit vertical-support-corrected LIPM'
    metadata['m1d_config_sha256']=sha256('configs/m1_coupled.yaml')
    metadata['scope']='Conditional coupled LIPM, passive bilateral contacts, contact-only robot budgets; not hardware'
    dump(out/'metadata.json',metadata)
    records=json.loads(Path(c['_capability']['candidates']).read_text())
    asset,model,checker=capability_setup(c['_capability']);entries=[]
    for id in c['candidates']:
        r=next(r for r in records if r['id']==id)
        for mode in c['experiment']['modes']:
            chart=SpatialChart(model,checker,r,c,mode=='mobile').build();envelopes=[]
            for radius in c['chart']['domains_m']:
                try:
                    e=chart.envelope(radius,c['contact'],c['robot']['torso_budget_nm']);e['status']='ACCEPTED';envelopes.append(e)
                except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    envelopes.append(dict(radius_m=radius,status='REJECTED',reason=str(exc)))
            chart.save(out/'charts'/f'{id}_{mode}.npz',envelopes)
            entry=dict(candidate=id,mode=mode,q0=r['q_rad'],base0=r['base_xy_yaw'],envelopes=envelopes,rejected=chart.rejected)
            entries.append(entry);print(id,mode,[(e['radius_m'],e.get('speed_m_s'),e.get('acceleration_m_s2'),e.get('reason')) for e in envelopes],flush=True)
    dump(out/'chart_inventory.json',entries)


STATE=None


def init_worker(c,out):
    global STATE
    asset,model,checker=capability_setup(c['_capability'])
    records=json.loads(Path(c['_capability']['candidates']).read_text())
    STATE=dict(c=c,out=Path(out),model=model,checker=checker,records=records,cache={})


def get_chart(candidate,mode,variant='main'):
    st=STATE;key=(candidate,mode,variant)
    if key in st['cache']:return st['cache'][key]
    c=variant_config(st['c'],variant);r=next(r for r in st['records'] if r['id']==candidate)
    chart=SpatialChart(st['model'],st['checker'],r,c,mode=='mobile')
    path=st['out']/'charts'/f'{candidate}_{mode}.npz'
    with np.load(path,allow_pickle=False) as data:
        chart.knots=data['knots_m'];chart.values=data['coordinates'];chart.slopes=data['slopes_per_m']
    chart.spline=CubicHermiteSpline(chart.knots,chart.values,chart.slopes)
    if variant=='main':envelopes=json.loads(Path(str(path)+'.json').read_text())['envelopes']
    else:
        envelopes=[]
        for radius in c['chart']['domains_m']:
            try:envelopes.append(dict(chart.envelope(radius,c['contact'],c['robot']['torso_budget_nm']),status='ACCEPTED'))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:envelopes.append(dict(radius_m=radius,status='REJECTED',reason=str(exc)))
    st['cache'][key]=(c,chart,[e for e in envelopes if e['status']=='ACCEPTED'])
    return st['cache'][key]


def compute(case):
    started=time.perf_counter();c,chart,envelopes=get_chart(case['candidate'],case['mode'],case['variant'])
    model=CoupledModel(c['_human'],c['contact'],c['experiment']['dt_s']);disturbance=Disturbance(**case['disturbance'])
    attempts=[];chosen=None;metric=None;status='NO_VERIFIED_RECOVERY'
    if case['method']=='offline':
        for e in envelopes:
            result=solve_offline(model,disturbance,case['horizon_s'],e,c['_profile']['delay_s'],c['experiment']['lp_time_limit_s'])
            log={k:v for k,v in result.items() if k not in ['state','acceleration']};log['envelope']=e
            if result['status']=='LP_FEASIBLE':
                attempt_path=str(Path(case['path_file']).with_suffix(f'.attempt{len(attempts)}.npz'))
                metrics=validate(model,chart,e,result['state'],result['acceleration'],disturbance,case['horizon_s'],c,attempt_path)
                log['path_file']=attempt_path
                log['validation']=metrics
                if metrics['valid'] and metrics['terminal_met']:
                    case=dict(case,path_file=attempt_path)
                    status='RECOVERED_OFFLINE';chosen=result;metric=metrics;attempts.append(log);break
                log['status']='POSTERIOR_REJECTED'
            attempts.append(log)
        if chosen is None:
            if not envelopes:status='INITIAL_TASK_OR_CHART_REJECTED'
            elif any(a['status']=='SOLVER_ERROR' for a in attempts):status='SOLVER_ERROR'
            elif any(a['status']=='POSTERIOR_REJECTED' for a in attempts):status='UNKNOWN_POSTERIOR_REJECTED'
            else:status='RESTRICTED_CLASS_INFEASIBLE'
    else:
        if not envelopes:status='INITIAL_TASK_OR_CHART_REJECTED'
        else:
            # Select from initial robot/constraint information only, no event or
            # impulse/horizon used to tune the chart or the fixed feedback gain.
            e=max(envelopes,key=lambda e:e['speed_m_s']*e['acceleration_m_s2']*(e['y_upper_m']-e['y_lower_m']))
            result=simulate_feedback(model,disturbance,case['horizon_s'],e,c['_profile']['delay_s'],c['feedback']['poles_s_inv'])
            metrics=validate(model,chart,e,result['state'],result['acceleration'],disturbance,case['horizon_s'],c,case['path_file'])
            status='RECOVERED_CAUSAL' if result['status']=='SIMULATED' and metrics['valid'] and metrics['terminal_met'] else 'NOT_RECOVERED_CAUSAL'
            metric=metrics;chosen=result
            attempts.append({k:v for k,v in result.items() if k not in ['state','acceleration']}|{'envelope':e})
    return dict(case,status=status,metrics=metric,attempts=attempts,elapsed_s=time.perf_counter()-started,
        terminal_definition='Human terminal box AND handle y/w near zero; offline additionally enforces all nominal terminal states zero',
        information='offline complete future disturbance known' if case['method']=='offline' else 'causal current observations and own queue only')


def cases_for(c,out,variant='main',candidates=None,horizons=None,impulses=None,modes=None,methods=('offline','causal')):
    cases=[]
    for id in candidates or c['candidates']:
        for mode in modes or c['experiment']['modes']:
            for h in horizons or c['experiment']['horizons_s']:
                for dv in impulses or c['experiment']['impulses_m_s']:
                    for method in methods:
                        name=f'{variant}_{id}_{mode}_{h:g}_{dv:+.5f}_{method}'
                        cases.append(dict(case_id=name,candidate=id,mode=mode,horizon_s=h,method=method,variant=variant,
                            disturbance=dict(kind='velocity_impulse',onset_s=0.,delta_v_m_s=float(dv)),
                            path_file=str(out/'paths'/(name+'.npz'))))
    return cases


def execute(c,out,cases,workers):
    (out/'cases').mkdir(exist_ok=True);(out/'paths').mkdir(exist_ok=True)
    pending=[x for x in cases if not (out/'cases'/(x['case_id']+'.json')).exists()]
    if workers==1:
        init_worker(c,out)
        for i,case in enumerate(pending):
            r=compute(case);dump(out/'cases'/(r['case_id']+'.json'),r)
            if i%10==0:print(i+1,'/',len(pending),r['case_id'],r['status'],flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(c,str(out))) as pool:
            for i,r in enumerate(pool.map(compute,pending,chunksize=1)):
                dump(out/'cases'/(r['case_id']+'.json'),r)
                if i%10==0:print(i+1,'/',len(pending),r['case_id'],r['status'],flush=True)
    return [json.loads((out/'cases'/(case['case_id']+'.json')).read_text()) for case in cases]


def refine_boundaries(c,out,workers):
    """First observed success-to-failure transition, never global infeasibility.

    Preserve nonmonotonic coarse outcomes and unknown solver/posterior cases.
    Bisection only localizes an observed transition; it is not a proof that every
    interior amplitude succeeds, particularly for causal feedback.
    """
    main=json.loads((out/'main.json').read_text());groups={};summaries=[]
    init_worker(c,out)
    success=lambda r:r['status'] in ['RECOVERED_OFFLINE','RECOVERED_CAUSAL']
    unknown=lambda r:r['status'] in ['SOLVER_ERROR','UNKNOWN_POSTERIOR_REJECTED','INITIAL_TASK_OR_CHART_REJECTED']
    for r in main:
        for sign in [-1,1]:
            dv=r['disturbance']['delta_v_m_s']
            if sign*dv>=0:groups.setdefault((r['candidate'],r['mode'],r['method'],r['horizon_s'],sign),[]).append(r)
    for (candidate,mode,method,h,sign),rows in groups.items():
        rows.sort(key=lambda r:abs(r['disturbance']['delta_v_m_s']))
        flags=[success(r) for r in rows];transition=next((i for i in range(1,len(rows)) if flags[i-1] and not flags[i]),None)
        info=dict(candidate=candidate,mode=mode,method=method,horizon_s=h,sign=sign,
            coarse_success=flags,coarse_nonmonotonic=any(flags[i] and not flags[i-1] for i in range(1,len(flags))),
            coarse_max_verified_m_s=max([abs(r['disturbance']['delta_v_m_s']) for r in rows if success(r)]+[0.]))
        if transition is None:
            summaries.append(dict(info,status='RIGHT_CENSORED' if flags[-1] else 'NO_OBSERVED_TRANSITION',
                lower_m_s=info['coarse_max_verified_m_s'],upper_m_s=None));continue
        low=abs(rows[transition-1]['disturbance']['delta_v_m_s']);high=abs(rows[transition]['disturbance']['delta_v_m_s'])
        status='LOCAL_TRANSITION';attempts=[]
        if unknown(rows[transition]):status='UNKNOWN_AT_COARSE_TRANSITION'
        else:
            while high-low>c['experiment']['boundary_resolution_m_s']+1e-10:
                mid=round((low+high)/2,8)
                case=cases_for(c,out,'main',[candidate],[h],[sign*mid],[mode],[method])[0]
                saved=out/'cases'/(case['case_id']+'.json')
                if saved.exists():r=json.loads(saved.read_text())
                else:r=compute(case);dump(saved,r)
                attempts.append(r['case_id'])
                if unknown(r):status='UNKNOWN_DURING_REFINEMENT';break
                if success(r):low=mid
                else:high=mid
        summaries.append(dict(info,status=status,lower_m_s=low,upper_m_s=high,refinement_cases=attempts,
            upper_meaning='restricted-class failure' if method=='offline' else 'this causal controller failure'))
        print(candidate,mode,method,h,sign,status,low,high,flush=True)
    dump(out/'boundaries.json',summaries)


def run(config_path,output,stage,workers):
    out=Path(output);c=load_config(config_path)
    if stage=='prepare':prepare(c,out);return
    if yaml.safe_load((out/'config.yaml').read_text())!=c:raise ValueError('configuration differs from saved experiment')
    dump(out/f'{stage}_provenance.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(),
        started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    if stage=='boundaries':refine_boundaries(c,out,workers);return
    if stage=='main':cases=cases_for(c,out)
    elif stage=='sensitivity':
        cases=[]
        for name in c['sensitivity']['variants']:
            cases+=cases_for(c,out,name,c['sensitivity']['candidates'],[c['sensitivity']['horizon_s']],c['sensitivity']['impulses_m_s'],['fixed'])
    elif stage=='future':
        cases=cases_for(c,out,candidates=['ik0024','ik0036'],horizons=[2.],impulses=[-.24,.24],modes=['fixed'])
        for case in cases:
            case['case_id']='future_'+case['case_id'];case['disturbance']['onset_s']=.32
            case['path_file']=str(out/'paths'/(case['case_id']+'.npz'))
    else:raise ValueError(stage)
    rows=execute(c,out,cases,workers);dump(out/(stage+'.json'),rows)
    print(stage,dict(Counter(r['status'] for r in rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/m1_coupled.yaml');p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['prepare','main','sensitivity','future','boundaries'],required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.config,a.output,a.stage,a.workers)
