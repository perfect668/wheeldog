"""Small unchanged-controller rerun; primary results are never overwritten."""
from pathlib import Path
import argparse
import json
import shutil
import concurrent.futures
import numpy as np
from h1_kinematics.assets import dump
from .common import load_config,provenance
from .evaluation import compute,initialize


def run(c,root,workers=4):
    root=Path(root);out=root/'repeatability'
    if out.exists():raise ValueError('repeatability diagnostic already exists')
    out.mkdir();(out/'charts').mkdir();(out/'paths').mkdir();(out/'cases').mkdir()
    for p in (root/'charts').glob('*.npz'):shutil.copyfile(p,out/'charts'/p.name)
    provenance(out,'repeatability')
    methods=json.loads((root/'selection.json').read_text())['methods'];ids={methods[k] for k in ['A_default','B_force','D_future']}
    original=json.loads((root/'closed_loop.json').read_text())
    selected=[r for r in original if r['candidate'] in ids and r['variant']=='nominal'
        and r['scenario']['id'] in ['pulse_n10','pulse_p10','pulse_n18','pulse_p18','pulse_n26','pulse_p26']]
    keys=['case_id','variant','scenario','candidate','controller','random_draw_candidate']
    plan=[{k:r[k] for k in keys} for r in selected];dump(out/'plan.json',plan);results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(c,out)) as pool:
        for r in pool.map(compute,plan,chunksize=1):
            dump(out/'cases'/(r['case_id']+'.json'),r);results.append(r)
    differences=[]
    for a,b in zip(selected,results):
        with np.load(a['path_file'],allow_pickle=False) as p,np.load(b['path_file'],allow_pickle=False) as q:
            x=p['observed_node_state'];y=q['observed_node_state'];n=min(len(x),len(y))
            state_error=float(np.max(abs(x[:n]-y[:n])))
            u=p['acceleration_m_s2'];v=q['acceleration_m_s2'];nu=min(len(u),len(v))
            command_error=float(np.max(abs(u[:nu]-v[:nu]))) if nu else 0.
        differences.append(dict(case_id=a['case_id'],controller=a['controller'],primary_sustained=a['sustained_recovered'],
            repeated_sustained=b['sustained_recovered'],decision_match=all(a[k]==b[k] for k in ['terminal_recovered','sustained_recovered','valid','observation_complete']),
            primary_intervals=a['completed_intervals'],repeat_intervals=b['completed_intervals'],
            maximum_common_node_state_error=state_error,maximum_common_acceleration_error_m_s2=command_error,
            primary_fallbacks=a['fallback_calls'],repeat_fallbacks=b['fallback_calls']))
    summary=dict(cases=len(results),decision_mismatches=sum(not d['decision_match'] for d in differences),comparisons=differences,
        interpretation='Unchanged model/controller/tuning and 4 workers; diagnostic, not an additional independent method test or replacement for main results')
    dump(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/m2_posture.yaml');p.add_argument('--output',default='results/m2a');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(load_config(a.config),a.output,a.workers)
