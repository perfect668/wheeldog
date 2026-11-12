"""All plots and tables are derived from saved, independently checked runs."""
from pathlib import Path
import csv
import json
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from h1_kinematics.assets import dump


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def statistics(rows,weights=None):
    w=np.ones(len(rows)) if weights is None else np.asarray(weights)
    def avg(key):return float(np.average([r[key] for r in rows],weights=w))
    complete=[r for r in rows if r['observation_complete']]
    recovered=[r for r in rows if r['sustained_recovered'] and r['recovery_time_s'] is not None]
    return dict(n_scenarios=float(sum(w)),terminal_rate=avg('terminal_recovered'),sustained_rate=avg('sustained_recovered'),
        human_only_terminal_rate=avg('human_only_terminal'),human_only_sustained_rate=avg('human_only_sustained'),
        completed_fraction=avg('observation_complete'),mean_peak_x_complete_m=float(np.mean([r['peak_human_x_m'] for r in complete])) if complete else None,
        mean_recovery_s_conditional=float(np.mean([r['recovery_time_s'] for r in recovered])) if recovered else None,
        max_recorded_per_hand_force_n=max(r['peak_per_hand_force_n'] for r in rows),
        mean_controller_wall_s=avg('control_wall_mean_s'),max_controller_wall_s=max(r['control_wall_max_s'] for r in rows))


def run(c,out):
    out=Path(out);fig=out/'figures';fig.mkdir(exist_ok=True)
    rows=json.loads((out/'closed_loop.json').read_text());sel=json.loads((out/'selection.json').read_text())
    # Post-hoc diagnostic only: separate human target from the additional
    # handle-return requirement. Primary saved decisions are never changed.
    diagnosis=[];h=c['_coupled']['_human']
    for r in rows:
        index=int(round(r['terminal_time_s']/c['control']['dt_s']))
        with np.load(r['path_file'],allow_pickle=False) as d:
            node=d['observed_node_state'];human=False;handle=False;sustained=False;endpoint=None
            if len(node)>index:
                endpoint=node[index].tolist();s=node[index]
                human=bool(abs(s[0])<=h['x_terminal_m'] and abs(s[1])<=h['v_terminal_m_s'])
                handle=bool(abs(s[2])<=c['_coupled']['feedback']['terminal_y_m'] and abs(s[3])<=c['_coupled']['feedback']['terminal_speed_m_s'])
                post=d['actual_state'][d['time_s']>=r['terminal_time_s']-1e-9]
                hs=np.vstack([post[:,:2],node[index:,:2]])
                sustained=bool(r['observation_complete'] and human and np.all(abs(hs[:,0])<=h['x_terminal_m']) and np.all(abs(hs[:,1])<=h['v_terminal_m_s']))
        r['human_only_terminal']=bool(r['valid'] and human);r['human_only_sustained']=bool(r['valid'] and sustained)
        reason=('RECOVERED' if r['sustained_recovered'] else ('INVALID_REPLAY' if not r['valid'] else ('EARLY_CONTROLLER_STOP' if not r['observation_complete'] else ('HUMAN_TERMINAL_BOX' if not human else ('HANDLE_RETURN_BOX' if not handle else 'OBSERVATION_ESCAPE')))))
        r['outcome_reason']=reason
        diagnosis.append(dict(case_id=r['case_id'],outcome_reason=reason,terminal_state=endpoint,
            human_terminal_box=human,handle_terminal_box=handle,human_only_terminal=r['human_only_terminal'],human_only_sustained=r['human_only_sustained']))
    dump(out/'outcome_diagnosis.json',diagnosis)
    metrics=sel['metrics'];scores=json.loads((out/'directional_scores.json').read_text())
    methods=dict(sel['methods']);methods['A_seeded_random']='seeded';methods['A_uniform_expected']='uniform'
    methods['B_capped_order_tie']=sel['B_contact_capped_order_tie']
    for method,candidate in sel['core_four_methods'].items():methods['core4_'+method]=candidate
    table=[]
    for group in ['nominal','nominal_matched_sensitivity','sensitivity','all']+list(c['evaluation']['variants']):
        if group=='nominal' and any(t['group']=='nominal' for t in table):continue
        group_rows=[r for r in rows if group=='all' or (group=='sensitivity' and r['variant']!='nominal') or r['variant']==group
            or (group=='nominal_matched_sensitivity' and r['variant']=='nominal' and r['scenario']['id'] in ['pulse_n18','pulse_p18','pulse_n26','pulse_p26'])]
        for controller in ['fixed','mpc']:
            rr=[r for r in group_rows if r['controller']==controller]
            for method,candidate in methods.items():
                use=rr if candidate=='uniform' else [r for r in rr if r['candidate']==(r['random_draw_candidate'] if candidate=='seeded' else candidate)]
                stat=statistics(use,np.full(len(use),1/len(c['candidates'])) if candidate=='uniform' else None)
                table.append(dict(group=group,controller=controller,method=method,selected=candidate,**stat))
    csv_write(out/'policy_summary.csv',table);dump(out/'policy_summary.json',table)
    csv_write(out/'candidate_metrics.csv',[{k:r[k] for k in ['candidate','R_negative_m_s','R_positive_m_s','recovery_min_m_s','recovery_mean_m_s','force_proxy_min_n','contact_capped_force_min_n','instantaneous_min_m_s','task_max_torque_ratio']} for r in metrics])
    keys=['case_id','candidate','controller','variant','terminal_recovered','sustained_recovered','human_only_terminal','human_only_sustained','outcome_reason','valid','observation_complete','peak_human_x_m','peak_per_hand_force_n','recovery_time_s','actual_max_torque_ratio','control_wall_mean_s','control_wall_max_s','missed_control_periods','fallback_calls']
    csv_write(out/'case_summary.csv',[{k:r[k] for k in keys}|dict(failures=';'.join(r['failures'])) for r in rows])
    candidate_rates=[]
    for m in metrics:
        rr=[r for r in rows if r['candidate']==m['candidate'] and r['variant']=='nominal' and r['controller']=='mpc']
        candidate_rates.append(dict(candidate=m['candidate'],score=m['recovery_min_m_s'],**statistics(rr)))
    csv_write(out/'candidate_rates.csv',candidate_rates)
    full_rates=[]
    for id in c['candidates']:
        for ctrl in ['fixed','mpc']:
            for group in ['nominal','sensitivity','all']:
                rr=[r for r in rows if r['candidate']==id and r['controller']==ctrl and (group=='all' or (group=='sensitivity' and r['variant']!='nominal') or r['variant']==group)]
                full_rates.append(dict(candidate=id,controller=ctrl,group=group,**statistics(rr)))
    csv_write(out/'all_candidate_rates.csv',full_rates)
    constraints=[]
    for id in c['candidates']:
        for ctrl in ['fixed','mpc']:
            rr=[r for r in rows if r['candidate']==id and r['controller']==ctrl and r['variant']=='nominal']
            def peak(key):return max(r['validation'].get(key,0) for r in rr)
            constraints.append(dict(candidate=id,controller=ctrl,
                max_joint_velocity_ratio=peak('max_joint_velocity_ratio'),max_joint_acceleration_ratio=peak('max_joint_acceleration_ratio'),
                max_contact_torque_ratio=max([r['actual_max_torque_ratio'] or 0 for r in rr]),max_contact_cap_ratio=peak('max_contact_cap_ratio'),
                max_handle_speed_ratio=max(r['validation'].get('peak_handle_speed_m_s',0)/r['envelope']['speed_m_s'] for r in rr),
                max_domain_ratio=max(r['validation'].get('peak_handle_displacement_m',0)/max(abs(r['envelope']['y_lower_m']),abs(r['envelope']['y_upper_m'])) for r in rr),
                fallback_calls=sum(r['fallback_calls'] for r in rr),completed=sum(r['observation_complete'] for r in rr)))
    csv_write(out/'constraints_by_candidate.csv',constraints)
    rank=spearmanr([r['score'] for r in candidate_rates],[r['sustained_rate'] for r in candidate_rates])
    command=[l for r in rows if r['controller']=='mpc' for l in r['command_logs']]
    iters=[a for s in scores for a in s['attempts']]
    failure=Counter(reason for r in rows for reason in r['failures'])
    summary=dict(selection=sel['methods'],runs=len(rows),terminal=sum(r['terminal_recovered'] for r in rows),
        sustained=sum(r['sustained_recovered'] for r in rows),failure_counts=dict(failure),
        outcome_reason_counts=dict(Counter(r['outcome_reason'] for r in rows)),
        score_attempts=len(iters),accepted_score_iterates=sum(a.get('validation',{}).get('valid',False) for a in iters),
        score_status_counts=dict(Counter(a['status'] for a in iters)),
        duplicate_initializations=sum(s['duplicate_of'] is not None for r in scores for s in r['seeds']),
        nominal_candidate_spearman_rho=float(rank.statistic) if np.isfinite(rank.statistic) else None,
        rank_caveat='Exploratory 8-candidate deterministic sample, no population significance claim',
        mpc_command_calls=len(command),mpc_fallback_calls=sum(l['fallback'] for l in command),
        mpc_qp_status=dict(Counter(l['qp_status'] for l in command)),
        mpc_pending_queue_constraint_violations=sum(l['pending_queue_constraint_violation'] for l in command),
        mpc_wall_quantiles_s={str(q):float(np.quantile([l['wall_s'] for l in command],q)) for q in [.5,.95,.99,1.]},
        mpc_over_control_period=sum(l['wall_s']>c['control']['dt_s'] for l in command),
        maximum_actual_replay_ode_error=max([r['actual_vs_replayed_ode_error'] or 0 for r in rows]),
        maximum_fk_position_error_m=max(r['validation'].get('max_handle_position_error_m',0) for r in rows),
        maximum_fk_rotation_error_rad=max(r['validation'].get('max_handle_rotation_error_rad',0) for r in rows),
        timing_caveat='4 CPU workers; no interruptible wall-time limit or simulated deadline jitter; max_iter identical',
        peak_caveat='Failed truncated trajectories have incomplete maxima; conditional completed-run metrics are selection-biased')
    positions=np.array([np.array(r['world_handle_poses'])[:,:3,3] for r in metrics])
    summary['initial_handle_position_spread_m']=float(np.max(np.linalg.norm(positions-positions[0],axis=2)))
    summary['B_contact_capped_argmax']=sel['B_contact_capped_argmax']
    summary['static_metric_spearman']={key:float(spearmanr([m[key] for m in metrics],[r['sustained_rate'] for r in candidate_rates]).statistic) for key in ['force_proxy_min_n','instantaneous_min_m_s']}
    # Paired differences use identical scenarios, never independent resampling.
    paired=[]
    index={(r['variant'],r['scenario']['id'],r['candidate'],r['controller']):r for r in rows}
    paired_peaks=[]
    for ctrl in ['fixed','mpc']:
        for group in ['nominal','all']:
            events=sorted({(r['variant'],r['scenario']['id']) for r in rows if group=='all' or r['variant']=='nominal'})
            common=[(v,s) for v,s in events if all(index[(v,s,id,ctrl)]['observation_complete'] and index[(v,s,id,ctrl)]['valid'] for id in sel['methods'].values())]
            for method,id in sel['methods'].items():
                rr=[index[(v,s,id,ctrl)] for v,s in common]
                paired_peaks.append(dict(group=group,controller=ctrl,method=method,n_common=len(rr),
                    mean_peak_x_m=float(np.mean([r['peak_human_x_m'] for r in rr])) if rr else None,
                    max_peak_x_m=max([r['peak_human_x_m'] for r in rr]) if rr else None))
    csv_write(out/'paired_complete_peak_summary.csv',paired_peaks)
    for ctrl in ['fixed','mpc']:
        for baseline in ['A_default','B_force','C_kinematic']:
            a=sel['methods'][baseline];d=sel['methods']['D_future'];pairs=[]
            for r in rows:
                if r['candidate']==d and r['controller']==ctrl:
                    b=index[(r['variant'],r['scenario']['id'],a,ctrl)]
                    pairs.append((r['sustained_recovered'],b['sustained_recovered']))
            paired.append(dict(controller=ctrl,comparison='D_future - '+baseline,n=len(pairs),
                wins=sum(x and not y for x,y in pairs),losses=sum(y and not x for x,y in pairs),
                sustained_rate_difference=float(np.mean([int(x)-int(y) for x,y in pairs]))))
    summary['paired_posture_ablation']=paired
    dump(out/'summary.json',summary)
    plt.rcParams.update({'font.size':9,'figure.dpi':140})
    def save(name):
        plt.gcf().suptitle('M2-A model simulation (uncalibrated robot/contact parameters)',fontsize=9)
        plt.tight_layout();plt.savefig(fig/(name+'.png'));plt.savefig(fig/(name+'.pdf'));plt.close()
    names=[r['candidate'] for r in metrics];x=np.arange(len(names));width=.25
    plt.figure(figsize=(9,3.5))
    for off,key,label in [(-width,'R_negative_m_s','-Y verified lower bound'),(0,'R_positive_m_s','+Y verified lower bound'),(width,'recovery_min_m_s','minimum score')]:
        plt.bar(x+off,[r[key] for r in metrics],width,label=label)
    plt.xticks(x,names);plt.ylabel('Offline impulse lower bound [m/s]');plt.legend(fontsize=8);save('directional_scores')
    plt.figure(figsize=(9,3.5))
    for i,start in enumerate(c['score']['starts']):
        bounds=[min(next(s for s in scores if s['candidate']==name and s['sign']==sign)['bounds_by_start'][start] for sign in [-1,1]) for name in names]
        plt.bar(x+(i-1)*width,bounds,width,label=start)
    plt.xticks(x,names);plt.ylabel('Min directional lower bound [m/s]');plt.legend();save('local_initialization')
    shown=['A_default','B_force','C_kinematic','D_future','A_uniform_expected']
    fig_,axes=plt.subplots(1,2,figsize=(11,3.8),sharey=True)
    for ax,group in zip(axes,['nominal','all']):
        for i,ctrl in enumerate(['fixed','mpc']):
            rr=[next(r for r in table if r['group']==group and r['method']==method and r['controller']==ctrl) for method in shown]
            ax.bar(np.arange(len(shown))+(i-.5)*.35,[r['sustained_rate']*100 for r in rr],.35,label=ctrl)
            ax.scatter(np.arange(len(shown))+(i-.5)*.35,[r['terminal_rate']*100 for r in rr],marker='_',s=180,color='black')
        ax.set_xticks(np.arange(len(shown)),[m.replace('_','\n',1) for m in shown]);ax.set_title(group+' holdout');ax.set_ylim(0,105);ax.legend()
    axes[0].set_ylabel('Recovery [%]; bars: sustained, marks: terminal');save('selection_controller_ablation')
    fig_,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    variants=['nominal_matched_sensitivity']+[v for v in c['evaluation']['variants'] if v!='nominal']
    for ax,ctrl in zip(axes,['fixed','mpc']):
        z=np.array([[next(r for r in table if r['group']==v and r['method']==method and r['controller']==ctrl)['sustained_rate'] for method in shown[:4]] for v in variants])
        im=ax.imshow(z,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(4),shown[:4],rotation=15);ax.set_yticks(range(len(variants)),['nominal (matched 4)']+variants[1:]);ax.set_title(ctrl)
        for i in range(z.shape[0]):
            for j in range(4):ax.text(j,i,f'{100*z[i,j]:.0f}',ha='center',va='center',color='white' if z[i,j]<.6 else 'black')
    fig_.colorbar(im,ax=axes.ravel().tolist(),label='Sustained fraction',fraction=.02)
    # tight_layout cannot place a shared colorbar reliably; explicit margins.
    fig_.suptitle('M2-A model simulation (uncalibrated robot/contact parameters)',fontsize=9)
    fig_.subplots_adjust(left=.16,right=.88,bottom=.16,top=.90,wspace=.08);fig_.savefig(fig/'sensitivity.png');fig_.savefig(fig/'sensitivity.pdf');plt.close(fig_)
    plt.figure(figsize=(6,3.5));plt.scatter([r['score'] for r in candidate_rates],[100*r['sustained_rate'] for r in candidate_rates])
    offsets=[(-8,12),(-38,12),(-25,-12),(-10,-12),(-20,10),(-20,-16),(-40,-12),(-5,12)]
    for r,offset in zip(candidate_rates,offsets):plt.annotate(r['candidate'],(r['score'],100*r['sustained_rate']),xytext=offset,textcoords='offset points',fontsize=8)
    plt.ylim(-8,85);plt.xlabel('Offline minimum lower bound [m/s]');plt.ylabel('Nominal causal MPC sustained [%]');save('score_closed_loop')
    # Both signs/controllers, no success-based trace filter. Original .26
    # representative retained; add the smallest predefined holdout pulse .10.
    for amplitude,filename in [('26','causal_traces'),('10','causal_traces_small')]:
        fig_,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
        for col,scenario in enumerate(['pulse_n'+amplitude,'pulse_p'+amplitude]):
            for method in ['A_default','D_future']:
                for ctrl,style in [('fixed',':'),('mpc','-')]:
                    rr=next(r for r in rows if r['variant']=='nominal' and r['scenario']['id']==scenario and r['candidate']==sel['methods'][method] and r['controller']==ctrl)
                    if rr['path_file']:
                        with np.load(rr['path_file'],allow_pickle=False) as d:
                            axes[0,col].plot(d['time_s'],d['actual_state'][:,0],style,label=f'{method}/{ctrl}')
                            axes[1,col].plot(d['time_s'],d['actual_per_hand_force_n'][:,0],style)
            axes[0,col].set_title(scenario);axes[0,col].axhspan(-.01,.01,alpha=.1,color='green')
            for ax in axes[:,col]:
                ax.axvspan(1.92,2.72,color='grey',alpha=.12);ax.set_xlim(0,2.72)
            axes[1,col].set_xlabel('Time [s]; grey: planned observation window')
        axes[0,0].set_ylabel('Actual human lateral x [m]');axes[1,0].set_ylabel('Left contact force [N]');axes[0,0].legend(fontsize=7);save(filename)
    plt.figure(figsize=(9,3.8))
    for i,(key,label) in enumerate([('max_joint_acceleration_ratio','joint acceleration'),('max_contact_torque_ratio','contact torque'),('max_contact_cap_ratio','contact force'),('max_domain_ratio','path domain')]):
        rr=[next(r for r in constraints if r['candidate']==id and r['controller']=='mpc') for id in names]
        plt.bar(x+(i-1.5)*.18,[r[key] for r in rr],.18,label=label)
    plt.axhline(1,color='black',linewidth=.6);plt.xticks(x,names);plt.ylabel('Peak recorded constraint usage, nominal MPC');plt.legend(fontsize=8);save('constraint_usage')
    print(json.dumps(summary,indent=2))
