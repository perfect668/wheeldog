"""Tables and scientific figures generated solely from saved M1-D records."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import subprocess
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from h1_kinematics.assets import dump
from .experiment import check_frozen,init_worker,get_chart
from .contact import CoupledModel


SUCCESS={'RECOVERED_OFFLINE','RECOVERED_CAUSAL'}
IDS=['ik0024','ik0036','ik0001','ik0010']
COLORS=dict(zip(IDS,['#b34725','#176dab','#668b31','#89569e']))


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def export(out):
    config=yaml.safe_load((out/'config.yaml').read_text());init_worker(config,out);initial=[]
    for candidate in IDS:
        _,chart,_=get_chart(candidate,'fixed');robot=chart.model
        budget=robot.budgets().copy();budget[robot.torso_indices]=config['robot']['torso_budget_nm']
        A=robot.contact_map(chart.q0,chart.b0);task=np.tile([0.,0.,config['contact']['vertical_per_hand_n']],2)
        tau=A@task;mask=np.isfinite(budget);model=CoupledModel(config['_human'],config['contact'],config['experiment']['dt_s'])
        lo,hi=-model.cap_y,model.cap_y;slope=A@np.tile([0.,1.,0.],2)
        for j in np.flatnonzero(mask):
            if abs(slope[j])>1e-12:
                l,h=sorted([(-budget[j]-tau[j])/slope[j],(budget[j]-tau[j])/slope[j]])
                lo=max(lo,l);hi=min(hi,h)
        initial.append(dict(candidate=candidate,q_rad=chart.q0.tolist(),base_xy_yaw=chart.b0.tolist(),
            world_handle_poses=np.array(chart.initial).tolist(),per_hand_current_force_n=[0,0,model.fz],
            contact_compensation_torque_nm=tau.tolist(),max_current_torque_ratio=float(max(abs(tau[mask])/budget[mask])),
            current_equal_per_hand_horizontal_force_interval_n=[lo,hi]))
    dump(out/'initial_tasks.json',initial)
    rows=[json.loads(p.read_text()) for p in sorted((out/'cases').glob('*.json'))]
    flat=[]
    for r in rows:
        row={k:r[k] for k in ['case_id','candidate','mode','horizon_s','method','variant','status','elapsed_s']}
        row.update(r['disturbance']);row.update(r.get('metrics') or {})
        row['path_file']=r['path_file'];row['attempt_count']=len(r['attempts'])
        for key in ['reasons','terminal_state','terminal_independent_human']:
            if key in row:row[key]=json.dumps(row[key])
        flat.append(row)
    write_csv(out/'cases.csv',flat)
    boundaries=json.loads((out/'boundaries.json').read_text());write_csv(out/'boundaries.csv',boundaries)
    main=json.loads((out/'main.json').read_text());sens=json.loads((out/'sensitivity.json').read_text())
    stats=[]
    for variant in ['main']+list(yaml.safe_load((out/'config.yaml').read_text())['sensitivity']['variants']):
        pool=main if variant=='main' else sens
        for candidate in IDS[:2]:
            for method in ['offline','causal']:
                selected=[r for r in pool if r['variant']==variant and r['candidate']==candidate and r['method']==method
                    and r['mode']=='fixed' and r['horizon_s']==1.6 and abs(r['disturbance']['delta_v_m_s']) in [.16,.24]]
                stats.append(dict(variant=variant,candidate=candidate,method=method,count=len(selected),
                    recovered=sum(r['status'] in SUCCESS for r in selected),statuses=dict(Counter(r['status'] for r in selected))))
    dump(out/'sensitivity_summary.json',stats);write_csv(out/'sensitivity_summary.csv',stats)
    accepted=[r for r in rows if r['status'] in SUCCESS]
    metrics=[r['metrics'] for r in accepted];limits={}
    for key in ['max_handle_position_error_m','max_handle_rotation_error_rad','dynamics_max_abs_error',
        'independent_ode_x_error_m','independent_ode_v_error_m_s','contact_force_model_max_error_n',
        'max_joint_velocity_ratio','max_joint_acceleration_ratio','max_base_velocity_ratio','max_base_acceleration_ratio',
        'max_torque_ratio','max_contact_cap_ratio','peak_human_x_m','peak_per_hand_lateral_force_n']:
        limits[key]=max(m[key] for m in metrics)
    reasons=Counter();attempts=Counter()
    for r in rows:
        reasons.update((r.get('metrics') or {}).get('reasons',[]))
        for a in r['attempts']:
            attempts[a['status']]+=1
            reasons.update(a.get('validation',{}).get('reasons',[]))
    future=json.loads((out/'future.json').read_text());prefix=[]
    for r in future:
        if not Path(r['path_file']).exists():continue
        with np.load(r['path_file'],allow_pickle=False) as data:
            mask=data['node_time_s']<r['disturbance']['onset_s']-1e-10
            prefix.append(dict(case_id=r['case_id'],status=r['status'],method=r['method'],
                max_pre_event_y_m=float(max(abs(data['node_state'][mask,2]))),
                max_pre_event_acceleration_m_s2=float(max(abs(data['acceleration_m_s2'][:int(round(.32/.04))])))))
    summary=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_integrity=check_frozen(),unique_cases=len(rows),all_statuses=dict(Counter(r['status'] for r in rows)),
        main_statuses=dict(Counter(r['status'] for r in main)),sensitivity_statuses=dict(Counter(r['status'] for r in sens)),
        accepted_trajectory_maxima=limits,attempt_statuses=dict(attempts),validation_reasons=dict(reasons),
        future_prefix=prefix,nonmonotonic_groups=sum(r['coarse_nonmonotonic'] for r in boundaries),
        total_case_wall_s=float(sum(r['elapsed_s'] for r in rows)),
        case_wall_median_s=float(np.median([r['elapsed_s'] for r in rows])),
        case_wall_max_s=float(max(r['elapsed_s'] for r in rows)),
        accepted_without_settled_recovery_time=sum(r['metrics']['recovery_time_s'] is None for r in accepted),
        interpretation='Conditional model results; restricted failures are not global impossibility')
    summary['direct_versions']={}
    for name in ['direct','direct_refined']:
        if not (out/(name+'.json')).exists():continue
        direct=json.loads((out/(name+'.json')).read_text())
        summary['direct_versions'][name]=dict(statuses=dict(Counter(r['status'] for r in direct)),
            attempts=sum(len(r['attempts']) for r in direct),
            valid_iterates=sum(a.get('validation',{}).get('valid',False) for r in direct for a in r['attempts']),
            rejection_reasons=dict(Counter(reason for r in direct for a in r['attempts'] for reason in a.get('validation',{}).get('reasons',[]))))
    if (out/'passive.json').exists():
        passive=json.loads((out/'passive.json').read_text())['cases']
        summary['stationary_handle_baseline']=dict(cases=len(passive),recovered=sum(r['recovered'] for r in passive))
    # A moving-base IK curve is not the union of all fixed/moving-base paths.
    summary['mobile_curve_failures_with_fixed_curve_success']=[]
    for r in main:
        if r['mode']!='fixed' or r['method']!='offline' or r['status'] not in SUCCESS:continue
        other=next(v for v in main if (v['candidate'],v['mode'],v['method'],v['horizon_s'],v['disturbance']['delta_v_m_s'])==
            (r['candidate'],'mobile',r['method'],r['horizon_s'],r['disturbance']['delta_v_m_s']))
        if other['status'] not in SUCCESS:summary['mobile_curve_failures_with_fixed_curve_success'].append(other['case_id'])
    dump(out/'summary.json',summary)
    return main,sens,boundaries,stats


def figures(out,main,sens,boundaries,stats):
    directory=out/'figures';directory.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(directory/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(directory/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    impulses=sorted(set(r['disturbance']['delta_v_m_s'] for r in main));horizons=[1.2,1.6,2.]
    cmap=ListedColormap(['#f2c8ba','#83bc9d','#d4b2db'])
    for ax,(mode,method) in zip(axes.flat,[(m,t) for m in ['fixed','mobile'] for t in ['offline','causal']]):
        image=np.zeros((12,len(impulses)));labels=[]
        for i,(candidate,h) in enumerate((c,h) for c in IDS for h in horizons):
            labels.append(f'{candidate} / {h:g}s')
            for j,dv in enumerate(impulses):
                r=next(r for r in main if (r['candidate'],r['mode'],r['method'],r['horizon_s'],r['disturbance']['delta_v_m_s'])==(candidate,mode,method,h,dv))
                image[i,j]=1 if r['status'] in SUCCESS else (2 if r['status'] in ['SOLVER_ERROR','UNKNOWN_POSTERIOR_REJECTED'] else 0)
        ax.imshow(image,aspect='auto',vmin=0,vmax=2,cmap=cmap);ax.grid(False)
        ax.set(xticks=range(len(impulses)),xticklabels=[f'{d:.2f}' for d in impulses],yticks=range(12),yticklabels=labels,
            title=f'{mode} base / {method}',xlabel='Initial lateral velocity impulse (m/s)')
    fig.suptitle('Coupled model simulation: green verified recovery; red method/class failure; purple unknown')
    fig.tight_layout(rect=(0,0,1,.97));save(fig,'recovery_grid')
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for ax,(mode,method) in zip(axes.flat,[(m,t) for m in ['fixed','mobile'] for t in ['offline','causal']]):
        for candidate in IDS:
            for sign in [-1,1]:
                rr=sorted([r for r in boundaries if (r['candidate'],r['mode'],r['method'],r['sign'])==(candidate,mode,method,sign)],key=lambda r:r['horizon_s'])
                ax.plot([r['horizon_s'] for r in rr],[r['lower_m_s'] for r in rr],
                    '-' if sign>0 else '--',marker='o',color=COLORS[candidate],label=candidate if sign>0 else None)
                for r in rr:
                    if r['upper_m_s'] is not None:ax.vlines(r['horizon_s'],r['lower_m_s'],r['upper_m_s'],color=COLORS[candidate],lw=2)
        ax.set(title=f'{mode} / {method}',xlabel='Horizon (s)',ylabel='Local verified impulse (m/s)',xticks=horizons)
    axes[0,0].legend(ncol=2);fig.suptitle('Local first observed transition: solid +Y, dashed -Y; short bars show brackets')
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'local_boundaries')
    fig,axes=plt.subplots(5,1,figsize=(10,11),sharex=True)
    selected=[r for r in main if r['candidate'] in IDS[:2] and r['mode']=='fixed' and r['horizon_s']==1.6
        and r['disturbance']['delta_v_m_s']==.16 and (r['method']=='causal' or r['candidate']=='ik0036')]
    for r in selected:
        if not Path(r['path_file']).exists():continue
        with np.load(r['path_file'],allow_pickle=False) as d:
            t=d['time_s'];state=d['state'];force=d['per_hand_lateral_force_n'];label=f"{r['candidate']} {r['method']} ({r['status']})"
            for ax,values in zip(axes,[state[:,0],state[:,1],force[:,0],d['handle_y_m'][:,0],d['handle_v_m_s'][:,0]]):
                ax.plot(t,values,color=COLORS[r['candidate']],ls='--' if r['method']=='offline' else '-',label=label)
    for ax,label in zip(axes,['Human x (m)','Human v (m/s)','Left actual Fy (N)','Left handle y (m)','Left handle speed (m/s)']):ax.set_ylabel(label)
    axes[0].legend(fontsize=7);axes[-1].set_xlabel('Time (s)');fig.suptitle('Fixed base, identical +0.16 m/s impulse; failed causal trace stops at failure')
    fig.tight_layout(rect=(0,0,1,.97));save(fig,'coupled_traces')
    variants=list(dict.fromkeys(r['variant'] for r in stats));fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
    for ax,method in zip(axes,['offline','causal']):
        for j,candidate in enumerate(IDS[:2]):
            values=[next(r['recovered']/r['count'] for r in stats if (r['method'],r['candidate'],r['variant'])==(method,candidate,v)) for v in variants]
            ax.bar(np.arange(len(variants))+(j-.5)*.36,values,width=.36,color=COLORS[candidate],label=candidate)
        ax.set(title=f'{method}, fixed base, H=1.6s',xticks=range(len(variants)),xticklabels=variants,ylabel='Verified fraction of four impulses',ylim=(0,1.05))
        ax.tick_params(axis='x',rotation=70);ax.legend()
    fig.suptitle('One-factor sensitivity: impulses +/-0.16 and +/-0.24 m/s (model assumptions)')
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'sensitivity')
    direct_file=out/('direct_refined.json' if (out/'direct_refined.json').exists() else 'direct.json')
    if direct_file.exists():
        direct=json.loads(direct_file.read_text());fig,ax=plt.subplots(figsize=(8,5));x=np.arange(len(direct))
        local=[];independent=[]
        for r in direct:
            b=next(b for b in boundaries if (b['candidate'],b['mode'],b['method'],b['horizon_s'],b['sign'])==(r['candidate'],'fixed','offline',1.6,r['sign']))
            local.append(b['lower_m_s']);independent.append(r['achieved_impulse_m_s'] if r['achieved_impulse_m_s'] is not None else np.nan)
        ax.bar(x-.18,local,.36,label='Spatial chart: local verified transition')
        ax.bar(x+.18,independent,.36,label='Direct joints: best verified lower bound')
        ax.set(xticks=x,xticklabels=[f"{r['candidate']} / sign {r['sign']:+d}" for r in direct],ylabel='Impulse amplitude (m/s)',title='Independent local trajectory methods; neither is a global boundary')
        ax.legend();fig.tight_layout();save(fig,'independent_joint_optimization')
    future=json.loads((out/'future.json').read_text());fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for i,candidate in enumerate(IDS[:2]):
        for r in future:
            if r['candidate']!=candidate or r['disturbance']['delta_v_m_s']!=.24 or not Path(r['path_file']).exists():continue
            with np.load(r['path_file'],allow_pickle=False) as d:
                for ax,j in zip(axes[i],[0,2]):ax.plot(d['time_s'],d['state'][:,j],label=r['method'])
        for ax,label in zip(axes[i],['Human x (m)','Handle y (m)']):
            ax.axvline(.32,color='k',ls=':',label='unannounced event');ax.set(title=candidate,ylabel=label,xlabel='Time (s)');ax.legend()
    fig.suptitle('Future +0.24 m/s impulse: offline knows it; causal interface has no event input')
    fig.tight_layout(rect=(0,0,1,.96));save(fig,'future_information')


def run(output):
    out=Path(output);data=export(out);figures(out,*data)
    print(json.dumps(json.loads((out/'summary.json').read_text()),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');run(p.parse_args().output)
