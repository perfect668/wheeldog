"""Acquisition evidence: outcomes, independent properties, real load/state curves."""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'artifacts/m2/dataset');p.add_argument('--output',type=Path,default=ROOT/'artifacts/m2/analysis');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
index=json.loads((a.dataset/'index.json').read_text());fs=json.loads((a.dataset/'features.json').read_text());g=fs['groups']
fig,axes=plt.subplots(1,3,figsize=(12,3.6))
for ax,(key,label) in zip(axes,[('K','K (Nm/rad)'),('D','D (Nm s/rad)'),('friction','Garment friction')]):
    vals=sorted(set(r[key] for r in index));good=[sum(r[key]==v and r['success'] for r in index) for v in vals];bad=[sum(r[key]==v and not r['success'] for r in index) for v in vals]
    x=np.arange(len(vals));ax.bar(x,good,label='Geometric completion');ax.bar(x,bad,bottom=good,label='Retained failure');ax.set_xticks(x,vals);ax.set(xlabel=label,ylabel='Episodes');ax.grid(axis='y',alpha=.2)
axes[-1].legend(fontsize=8);fig.tight_layout();fig.savefig(a.output/'parameter_coverage.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
# Matching alignment/action seed is not promised. Curves demonstrate real runs,
# not a controlled identification benchmark.
selected=[]
for k,d in [(4.,.5),(4.,2.),(20.,.5),(20.,2.)]:
    candidates=[r for r in index if r['K']==k and r['D']==d and r['friction']==.3 and r['profile']=='baseline']
    if candidates:selected.append(candidates[0])
for record in selected:
    data=np.load(a.dataset/record['file']);x=data['state'];t=data['time'];label=f'K={record["K"]:g}, D={record["D"]:g}, mu={record["friction"]:g}'
    axes[0].plot(t,x[:,g['human_q'][0]+1]-data['condition'][10],label=label)
    axes[1].plot(t,x[:,g['progress'][0]]*1000,label=label)
    load=x[:,slice(*g['load'])].reshape(-1,2,6)
    axes[2].plot(t,np.linalg.norm(load[:,:,:3],axis=-1).mean(1),label=label)
for ax,label in zip(axes,['Shoulder Y minus reference (rad)','Cuff axial from wrist (mm)','Mean wrist net external force (N)']):
    ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[-1].set_xlabel('Simulation time (s)');fig.tight_layout();fig.savefig(a.output/'interaction_responses.png',dpi=170);fig.savefig(a.output/'interaction_responses.pdf');plt.close(fig)
summary=dict(episodes=len(index),successes=sum(r['success'] for r in index),failures=sum(not r['success'] for r in index),
    termination_counts=dict(collections.Counter(r['termination_reason'] for r in index)),
    valid_states=sum(r['valid_states'] for r in index),total_simulated_s=sum(r['simulated_seconds'] for r in index),
    combinations=len(set(r['parameter_group'] for r in index)),
    independent_values={key:sorted(set(r[key] for r in index)) for key in ['K','D','friction']},
    max_no_cloth_force_residual_N=max(r['max_no_cloth_residual_force_N'] for r in index),
    max_no_cloth_torque_residual_Nm=max(r['max_no_cloth_residual_torque_Nm'] for r in index),
    resource_sampling='setup_logs/m2/resources.jsonl; 10 s interval, finite monitored window')
resources=ROOT/'setup_logs/m2/resources.jsonl'
if resources.exists():
    rows=[json.loads(r) for r in resources.read_text().splitlines()];memory=[];gpu=[];rss=[];cpu=[];last={}
    for row in rows:
        m,u,power=map(float,row['gpu_MiB_percent_W'].split(','));memory.append(m);gpu.append(u)
        for proc in row['processes']:
            rss.append(proc['rss_bytes']);key=proc['pid'];now=row['wall_time_ns']/1e9
            if key in last:
                old_t,old_ticks=last[key];cpu.append((proc['cpu_ticks']-old_ticks)/100/(now-old_t)*100)
            last[key]=(now,proc['cpu_ticks'])
    summary['resources']=dict(samples=len(rows),gpu_peak_MiB=max(memory),gpu_mean_utilization_percent=float(np.mean(gpu)),
        process_peak_RSS_GiB=max(rss)/2**30 if rss else None,process_peak_CPU_percent=max(cpu) if cpu else None,
        interval_s=10,cpu_clock_ticks_per_second=100,limitation='sampled peaks, not instantaneous full-run guarantees; may include offline model process')
(a.output/'collection_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
