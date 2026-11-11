"""Audit main sweep and bisect positive oracle boundaries across dt (CPU only)."""
import argparse
from dataclasses import replace
import csv
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from recoverability.model import Model
from recoverability.actuator import Actuator
from recoverability.disturbance import Disturbance
from recoverability.solver import solve_oracle
from recoverability.experiment import Dataset, provenance
from recoverability.plots import load_rows
from recoverability.metrics import validate


def audit_sweep(directory):
    all_rows=load_rows(directory)
    rows=[r for r in all_rows if r['family']=='sweep' and r['method']=='oracle']
    lookup={(r['actuator'],float(r['horizon_s']),round(float(r['delta_v_m_s']),8)):r['status'] for r in rows}
    names=sorted(set(r['actuator'] for r in rows))
    hs=sorted(set(float(r['horizon_s']) for r in rows))
    vs=sorted(set(round(float(r['delta_v_m_s']),8) for r in rows))
    symmetry=sum(lookup.get((name,h,v)) != lookup.get((name,h,-v)) for name in names for h in hs for v in vs if v>0)
    horizon_anomalies=sum(lookup[name,hs[j],v]=='RECOVERABLE' and lookup[name,hs[j+1],v]!='RECOVERABLE'
                         for name in names for j in range(len(hs)-1) for v in vs)
    capability_anomalies=sum(lookup[name,h,v]=='RECOVERABLE' and lookup['A_fast',h,v]!='RECOVERABLE'
                             for name in names if name!='A_fast' for h in hs for v in vs)
    errors=sum(r['status']=='SOLVER_ERROR' for r in rows)
    accepted=[r for r in rows if r['status']=='RECOVERABLE']
    result={'oracle_sweep_cases':len(rows),'symmetry_disagreements':symmetry,
            'observed_horizon_anomalies':horizon_anomalies,'observed_capability_anomalies':capability_anomalies,
            'solver_errors':errors,'posterior_validated_recoverable_cases':len(accepted),
            'all_accepted_pass_checks':all(r['valid_path_dynamics']=='True' and r['terminal_met']=='True' for r in accepted)}
    for k in ('dynamics_x_error_m','dynamics_v_error_m_s','actuator_error_N','increment_violation_N','x_path_violation_m','v_path_violation_m_s'):
        result['max_'+k]=max(float(r[k]) for r in accepted)
    replay_errors=[]
    with np.load(Path(directory)/'trajectories.npz',allow_pickle=False) as data:
        saved=[json.loads(r) for r in data['rows_json']]
        if len(saved)!=len(all_rows):replay_errors.append('CSV/NPZ row count mismatch')
        for row,packed in zip(all_rows,saved):
            key=row['case_id']
            if row['status']!=packed['status'] or row['git_commit']!=packed['git_commit']:
                replay_errors.append(key+': metadata mismatch')
            state,force,u=(data[key+suffix] for suffix in ('_state','_force','_command'))
            if len(state):
                m=Model(**json.loads(row['model_json']));a=Actuator(**json.loads(row['actuator_json']))
                d=Disturbance(kind=row['kind'],onset_s=float(row['onset_s']),
                              delta_v_m_s=float(row['delta_v_m_s']),force_N=float(row['force_N']),duration_s=float(row['duration_s']))
                metrics=validate(m,a,d,state,force,u)
                good=metrics['valid_path_dynamics'] and metrics['terminal_met']
                accepted_status=row['status'] in ('RECOVERABLE','RECOVERED')
                if good!=accepted_status:replay_errors.append(key+': replay classification mismatch')
            elif row['status'] not in ('INFEASIBLE','SOLVER_ERROR'):
                replay_errors.append(key+': missing trajectory')
    result['raw_replay_cases']=len(all_rows)
    result['raw_replay_mismatches']=len(replay_errors)
    result['raw_replay_errors']=replay_errors
    return result


def plot_convergence(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (output/'convergence.csv').open() as f: rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,h in zip(axes,(.8,2.0)):
        for name in dict.fromkeys(r['actuator'] for r in rows):
            group=sorted([r for r in rows if r['actuator']==name and float(r['horizon_s'])==h],key=lambda r:float(r['dt_s']))
            x=[float(r['dt_s']) for r in group]
            lo=np.array([float(r['confirmed_low_m_s']) for r in group]);hi=np.array([float(r['confirmed_high_m_s']) for r in group])
            ax.errorbar(x,(lo+hi)/2,yerr=(hi-lo)/2,marker='o',label=name,capsize=3)
        ax.set(xlabel='dt (s)',title=f'H={h:g} s');ax.grid(alpha=.2)
    axes[0].set_ylabel('Positive oracle boundary Δv (m/s)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Toy simulation · time-step sensitivity\nerror bars = bisection brackets, not confidence intervals')
    for ext in ('png','pdf'):fig.savefig(output/f'time_step_sensitivity.{ext}',bbox_inches='tight',dpi=160)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/exp0.yaml'))
    parser.add_argument('--input',type=Path,default=Path('results/exp0'))
    parser.add_argument('--output',type=Path,default=Path('results/validation'))
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    if args.plots_only:
        plot_convergence(args.output);return
    if args.output.exists() and any(args.output.iterdir()):parser.error('output must be empty')
    config=yaml.safe_load(args.config.read_text())
    meta=provenance(config,args.seed)
    meta['validation_settings']={'dt_s':[.05,.02,.01,.005],'horizons_s':[.8,2.0],
                                  'bisection_tolerance_m_s':.0005,'initial_bracket_m_s':[0,.8]}
    dataset=Dataset(meta);boundaries=[]
    for dt in meta['validation_settings']['dt_s']:
        model=replace(Model(**config['model']),dt_s=dt)
        for name,parameters in config['actuators'].items():
            actuator=Actuator(**parameters)
            for horizon in (.8,2.0):
                low,high=0.0,.8;errors=0
                def evaluate(v):
                    d=Disturbance(delta_v_m_s=v)
                    result=solve_oracle(model,actuator,d,horizon,time_limit_s=config['solver']['time_limit_s'])
                    dataset.add(model,actuator,name,d,horizon,'oracle',result,family='bisection')
                    return result.status
                low_status,high_status=evaluate(low),evaluate(high)
                if low_status!='RECOVERABLE' or high_status!='INFEASIBLE':
                    raise RuntimeError(f'Invalid initial bracket: {low_status}, {high_status}')
                while high-low > .0005:
                    mid=(low+high)/2
                    status=evaluate(mid)
                    if status=='RECOVERABLE':low=mid
                    elif status=='INFEASIBLE':high=mid
                    else:errors+=1;break  # UNKNOWN: retain wide bracket, never convert error to failure
                boundaries.append({'actuator':name,'dt_s':dt,'horizon_s':horizon,
                                   'confirmed_low_m_s':low,'confirmed_high_m_s':high,
                                   'bracket_width_m_s':high-low,'solver_errors':errors,
                                   'seed':args.seed,'git_commit':meta['git_commit']})
        print(f'dt={dt:g}s completed',flush=True)
    dataset.save(args.output)
    with (args.output/'convergence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(boundaries[0]));w.writeheader();w.writerows(boundaries)
    audit=audit_sweep(args.input)
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    plot_convergence(args.output)
    print(json.dumps(audit,indent=2),flush=True)
    if audit['solver_errors'] or audit['raw_replay_mismatches'] or any(r['solver_errors'] for r in boundaries):
        sys.exit(2)


if __name__=='__main__':main()
