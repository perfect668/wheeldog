"""Independent saved-data checks and denser replay; never changes main decisions."""
from pathlib import Path
import argparse
import json
import numpy as np
from h1_kinematics.assets import dump
from h1_coupled.contact import CoupledModel
from h1_coupled.direct import replay
from h1_coupled.validation import validate
from .common import Context,load_config,variant,frozen,provenance
from .evaluation import ScheduledDisturbance,recovery_metrics


def run(c,out):
    out=Path(out);provenance(out,'audit');ctx=Context(c,out);scores=json.loads((out/'directional_scores.json').read_text())
    rows=json.loads((out/'closed_loop.json').read_text());sel=json.loads((out/'selection.json').read_text())
    dense=out/'dense_audit';dense.mkdir(exist_ok=True);disagreements=[];checked=0
    for r in rows:
        cc=variant(c,r['variant']);m=CoupledModel(cc['_human'],cc['contact'],c['control']['dt_s'])
        with np.load(r['path_file'],allow_pickle=False) as d:
            s=d['actual_state'];nodes=d['observed_node_state'];t=d['time_s']
            tt=np.r_[t,np.arange(len(nodes))*m.dt];ss=np.vstack([s,nodes]);order=np.argsort(tt,kind='stable')
            last=max([t for t,v in r['scenario']['events']]+[0.])
            metric=recovery_metrics(tt[order],ss[order],nodes,r['terminal_time_s'],r['observation_end_s'],last,m,cc,r['valid'])
            if any(metric[k]!=r[k] for k in ['terminal_recovered','sustained_recovered','observation_complete']):disagreements.append(r['case_id']+': outcome')
            if r['valid'] and len(s):
                if max(abs(ss[:,0]))>m.human.x_safe_m+1e-7 or max(abs(ss[:,1]))>m.human.v_abs_max_m_s+1e-7:disagreements.append(r['case_id']+': human')
                if np.max(np.hypot(d['actual_per_hand_force_n'],m.fz))>m.contact['resultant_cap_per_hand_n']+1e-5:disagreements.append(r['case_id']+': force')
                if not np.isfinite(d['actual_torque_nm']).all():disagreements.append(r['case_id']+': nonfinite torque')
            checked+=1
    score_audit=[]
    for r in scores:
        if not r['path_file']:continue
        cc=variant(c,'nominal');cc['direct'].update(c['score']);cc['experiment']['validation_dt_s']=.0025
        cc['experiment']['collision_stride']=4
        chart,_,_=ctx.chart(r['candidate'],'speed_weighted',cc);m=CoupledModel(cc['_human'],cc['contact'],c['control']['dt_s'])
        best=next(a for a in r['attempts'] if a['start']==r['best_start'] and a.get('validation',{}).get('valid')
            and abs(a['impulse_m_s']-r['lower_bound_m_s'])<1e-12)
        with np.load(best['iterate_file'],allow_pickle=False) as d:
            v=replay(chart,m,d['states'],d['acceleration'],r['lower_bound_m_s'],r['sign'],cc,dense/f"score_{r['candidate']}_{r['sign']:+d}.npz")
        score_audit.append(dict(candidate=r['candidate'],sign=r['sign'],**v))
        print('dense score',r['candidate'],r['sign'],v['valid'],v['reasons'],flush=True)
    loop_audit=[]
    for r in rows:
        if r['variant']!='nominal' or r['scenario']['id'] not in ['pulse_n26','pulse_p26'] or r['candidate'] not in {sel['methods']['A_default'],sel['methods']['D_future']}:continue
        cc=variant(c,'nominal');cc['experiment']['validation_dt_s']=.0025;cc['experiment']['collision_stride']=4
        chart,e,_,_=ctx.controller_chart(r['candidate'],cc);m=CoupledModel(cc['_human'],cc['contact'],c['control']['dt_s'])
        with np.load(r['path_file'],allow_pickle=False) as d:
            if 'node_state' not in d:continue
            v=validate(m,chart,e,d['node_state'],d['acceleration_m_s2'],ScheduledDisturbance(r['scenario']['initial_v_m_s'],r['scenario']['events']),r['observation_end_s'],cc,dense/(r['case_id']+'.npz'))
        loop_audit.append(dict(case_id=r['case_id'],main_valid=r['valid'],**v))
    result=dict(saved_cases_checked=checked,saved_outcome_or_constraint_disagreements=disagreements,
        dense_dt_s=.0025,mesh_dt_s=.01,dense_score_audit=score_audit,dense_loop_audit=loop_audit,
        old_frozen=frozen(),scope='Supplementary replay resolution check, not continuous-time or invariant-set proof')
    dump(out/'audit.json',result)
    if disagreements or any(not r['valid'] for r in score_audit):raise ValueError('Supplementary audit found disagreements; inspect audit.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/m2_posture.yaml');p.add_argument('--output',default='results/m2a')
    a=p.parse_args();run(load_config(a.config),a.output)
