"""Summarize true 240 Hz e088 reruns without deleting or relabeling spikes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
JOINTS=['shoulder X','shoulder Y','shoulder Z','elbow']


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def compact_excerpt(row):
    """Review excerpt only; complete 240 Hz rings/patch arrays stay in source."""
    value={k:v for k,v in row.items() if k!='cuff'}
    c=row['cuff']
    if c is None:
        value['cuff']=None
    else:
        ring=np.array(c['ring_m'])
        value['cuff']={k:v for k,v in c.items() if k not in ('ring_m','opening','attachments')}
        value['cuff']['opening']={k:v for k,v in c['opening'].items() if k!='polygon'}
        value['cuff']['eight_ring_nodes_m']=ring[np.linspace(0,len(ring)-1,8,dtype=int)]
        value['cuff']['attachments']={s:{'maximum_error_m':c['attachments'][s]['maximum_error_m'],
            'attached_centroid_m':np.mean(c['attachments'][s]['attached_nodes_m'],axis=0),
            'expected_centroid_m':np.mean(c['attachments'][s]['expected_nodes_m'],axis=0),
            'node_count':len(c['attachments'][s]['indices'])} for s in 'LR'}
    return value


def analyze(path,out):
    rows=load_rows(path/'physics_240hz.jsonl')
    bins=load_rows(path/'control_20hz_aggregates.jsonl')
    time=np.array([r['simulation_time_s'] for r in rows]);dt=1/240
    raw=np.array([[r['solver_state']['raw_child_frame_N_Nm'][s] for s in 'LR'] for r in rows])
    external=np.array([[r['estimated_load']['external_world_N_Nm'][s] for s in 'LR'] for r in rows])
    support=np.array([[r['estimated_load']['inertial_support_world_N_Nm'][s] for s in 'LR'] for r in rows])
    q=np.array([r['human']['q_rad'] for r in rows]);dq=np.array([r['human']['dq_rad_s'] for r in rows])
    explicit=np.array([r['human']['explicit_effort_Nm'] for r in rows])
    cuff=np.array([r['cuff']['opening']['center'] if r['cuff'] else [np.nan]*3 for r in rows])
    progress=np.array([r['cuff']['axial_from_wrist_m'] if r['cuff'] else np.nan for r in rows])
    grasp=np.array([[r['cuff']['attachments'][s]['maximum_error_m'] if r['cuff'] else np.nan for s in 'LR'] for r in rows])
    norm=np.linalg.norm(external[:,:,:3],axis=-1);raw_norm=np.linalg.norm(raw[:,:,:3],axis=-1)
    peak=int(np.argmax(norm.max(axis=1)));peak_s=float(time[peak])
    control_ids=np.array([r['control_step'] for r in rows])
    # Force-peak wrenches are recorded online. Torque-norm peaks are derived
    # independently here from every native physical sample in each same bin.
    control_torque_peak=[];control_torque_peak_time=[]
    for b in bins:
        ids=np.flatnonzero(control_ids==b['control_step'])
        torque_norm=np.linalg.norm(external[ids,:,3:],axis=-1)
        offsets=torque_norm.argmax(axis=0)
        control_torque_peak.append(torque_norm.max(axis=0))
        control_torque_peak_time.append(time[ids[offsets]])
    simultaneous=external[peak,:,:3]
    cosine=float(simultaneous[0]@simultaneous[1]/max(np.prod(np.linalg.norm(simultaneous,axis=1)),1e-12))
    events={}
    for threshold in [20,50,100]:
        mask=norm.max(axis=1)>threshold
        segments=[]
        starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]])
        stops=np.flatnonzero(mask&~np.r_[mask[1:],False])+1
        for start,stop in zip(starts,stops):
            segments.append(dict(start_simulation_s=float(time[start]),end_sample_simulation_s=float(time[stop-1]),
                sample_count=int(stop-start),resolved_duration_s=float((stop-start)*dt),
                peak_force_N=float(norm[start:stop].max()),
                vector_impulse_L_Ns=external[start:stop,0,:3].sum(0)*dt,
                vector_impulse_R_Ns=external[start:stop,1,:3].sum(0)*dt))
        events[str(threshold)]=dict(total_samples=int(mask.sum()),total_resolved_duration_s=float(mask.sum()*dt),segments=segments)
    before=max(0,peak-1);after=min(len(rows)-1,peak+1)
    window=np.flatnonzero(np.abs(time-peak_s)<=.15)
    whole=np.flatnonzero((time>=10.7)&(time<=11.7))
    if not len(whole):whole=window
    original_center=int(np.argmin(np.abs(time-11.20833391789347)))
    selected_bin=next(b for b in bins if b['control_step']==rows[peak]['control_step'])
    result=json.loads((path/'result.json').read_text())
    summary=dict(episode_id=path.name,rows=len(rows),simulation_s=float(time[-1]-time[0]),
        successful=result['success'],termination_reason=result['termination_reason'],
        peak_simulation_s=peak_s,peak_state=rows[peak]['state'],
        peak_estimated_force_N=norm[peak],peak_raw_force_N=raw_norm[peak],
        peak_inertial_support_force_N=np.linalg.norm(support[peak,:,:3],axis=-1),
        simultaneous_force_cosine=cosine,simultaneous_world_force_sum_N=simultaneous.sum(0),
        peak_raw_child_wrench_N_Nm=raw[peak],peak_estimated_world_wrench_N_Nm=external[peak],
        peak_grasp_error_m=grasp[peak],peak_human_q_rad=q[peak],peak_human_dq_rad_s=dq[peak],
        peak_step_human_delta_dq_rad_s=dq[peak]-dq[before],
        next_step_human_delta_dq_rad_s=dq[after]-dq[peak],
        q_range_in_original_spike_window_rad=np.ptp(q[whole],axis=0),
        dq_range_in_original_spike_window_rad_s=np.ptp(dq[whole],axis=0),
        original_peak_time_nearest_sample_s=float(time[original_center]),
        force_near_original_peak_time_N=norm[original_center],
        resolved_events=events,peak_control_bin=selected_bin,
        max_attachment_error_m=np.nanmax(grasp,axis=0),
        source_sha256=hashlib.sha256((path/'physics_240hz.jsonl').read_bytes()).hexdigest(),
        interpretation='Native reaction and compensated net wrist load remain separate; neither isolates cloth-human contact. Duration is resolved physical samples times 1/240 s. No causal contact-source isolation was performed.')
    out.mkdir(parents=True,exist_ok=True)
    (out/(path.name+'_peak_excerpt.json')).write_text(json.dumps(dict(summary=summary,
        excerpt='Every 240 Hz sample in peak +/-150 ms; cuff ring reduced to eight nodes and attachment centroids for review only. Full arrays remain in server source JSONL.',
        rows=[compact_excerpt(rows[i]) for i in window]),indent=2,default=lambda v:v.tolist()))
    np.savez_compressed(out/(path.name+'_traces.npz'),time=time,raw_child_N_Nm=raw,
        estimated_world_N_Nm=external,inertial_support_world_N_Nm=support,human_q_rad=q,
        human_dq_rad_s=dq,human_explicit_Nm=explicit,cuff_center_m=cuff,progress_m=progress,attachment_error_m=grasp,
        control_start_s=np.array([b['first_simulation_s'] for b in bins]),
        control_end_exclusive_s=np.array([b['interval_end_exclusive_s'] for b in bins]),
        control_sample_count=np.array([b['sample_count'] for b in bins]),
        control_complete=np.array([b['complete'] for b in bins]),
        control_mean_signed_world_N_Nm=np.array([[b['estimated_world'][s]['mean_signed_wrench_N_Nm'] for s in 'LR'] for b in bins]),
        control_impulse_world_Ns_Nms=np.array([[b['estimated_world'][s]['vector_impulse_Ns_Nms'] for s in 'LR'] for b in bins]),
        control_mean_force_norm_N=np.array([[b['estimated_world'][s]['mean_force_norm_N'] for s in 'LR'] for b in bins]),
        control_peak_force_norm_N=np.array([[b['estimated_world'][s]['peak_force_norm_N'] for s in 'LR'] for b in bins]),
        control_peak_wrench_world_N_Nm=np.array([[b['estimated_world'][s]['peak_wrench_N_Nm'] for s in 'LR'] for b in bins]),
        control_peak_simulation_s=np.array([[b['estimated_world'][s]['peak_simulation_s'] for s in 'LR'] for b in bins]),
        control_peak_torque_norm_Nm=np.array(control_torque_peak),
        control_peak_torque_simulation_s=np.array(control_torque_peak_time))
    fig,axes=plt.subplots(3,2,figsize=(13,10))
    for i,s in enumerate('LR'):
        axes[0,0].plot(time,raw_norm[:,i],label=s,lw=.8)
        axes[0,1].plot(time,norm[:,i],label=s,lw=.8)
        axes[1,1].plot(time[window],norm[window,i],label=s,marker='.',ms=2)
        axes[2,1].plot(time[window],grasp[window,i]*1000,label=s)
    for j,name in enumerate(JOINTS):
        axes[1,0].plot(time[window],q[window,j],label=name)
        axes[2,0].plot(time[window],dq[window,j],label=name)
    for ax,label in zip(axes.flat,['Raw incoming force norm (N)','Estimated net wrist force norm (N)',
        'Human q (rad), peak +/- 150 ms','Estimated force (N), peak +/- 150 ms',
        'Human dq (rad/s), peak +/- 150 ms','Attachment maximum error (mm), peak +/- 150 ms']):
        ax.set(xlabel='Simulation time (s)',ylabel=label);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(path.name+' — actual 240 Hz physical-step observations')
    fig.tight_layout();fig.savefig(out/(path.name+'_peak.png'),dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for i,s in enumerate('LR'):
        axes[0,0].plot(time[whole],norm[whole,i],label=s,lw=.9)
        axes[0,1].plot(time[whole],raw_norm[whole,i],label=s,lw=.9)
    for j,name in enumerate(JOINTS):
        axes[1,0].plot(time[whole],q[whole,j],label=name)
        axes[1,1].plot(time[whole],dq[whole,j],label=name)
    for ax,label in zip(axes.flat,['Estimated net wrist force (N)','Raw incoming force (N)','Human q (rad)','Human dq (rad/s)']):
        ax.axvline(11.20833391789347,color='gray',ls=':',label='Original e088 peak time')
        ax.set(xlabel='Simulation time (s)',ylabel=label);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle(path.name+' — retained original e088 time window')
    fig.tight_layout();fig.savefig(out/(path.name+'_original_window.png'),dpi=170);plt.close(fig)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'artifacts/m21/highfreq/e088_directed')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/m21/highfreq/analysis')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for path in sorted(a.input.iterdir()):
        if (path/'high_frequency_metadata.json').exists() and (path/'result.json').exists():
            summaries.append(analyze(path,a.output))
    if not summaries:raise RuntimeError('No completed high-frequency reruns found')
    (a.output/'summary.json').write_text(json.dumps(dict(reruns=summaries,
        role='Directed regression diagnostics; not included in model selection or original test metrics'),indent=2,default=lambda v:v.tolist()))
    print(json.dumps([{k:s[k] for k in ['episode_id','rows','successful','termination_reason','peak_simulation_s','peak_estimated_force_N']} for s in summaries],default=lambda v:v.tolist()))


if __name__=='__main__':main()
