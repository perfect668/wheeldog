"""Read actual task episodes; compact numeric review and truthful response plots."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('episodes',nargs='+',type=Path);p.add_argument('--output',type=Path,default=ROOT/'artifacts/m11/analysis');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True);joint_fig,joint_axes=plt.subplots(2,2,figsize=(10,6),sharex=True);summary=[]
for run in a.episodes:
    rows=[json.loads(s) for s in (run/'samples.jsonl').read_text().splitlines()];meta=json.loads((run/'metadata.json').read_text());result=json.loads((run/'result.json').read_text())
    states=[r['solver_state']['value'] for r in rows];t=np.array([r['simulation_time_s'] for r in rows]);t-=t[0]
    steps=np.array([r['physics_step'] for r in rows]);assert np.all(np.diff(steps)>0)
    np.testing.assert_allclose(np.diff(t),np.diff(steps)/240,atol=2e-6)
    q=np.array([s['human_q_rad'] for s in states]);dq=np.array([s['robot_dq'] for s in states]);reference=np.array(meta['config']['human']['reference'])
    precontrol=np.array([[s['end_effector_tracking_error_m'][side] for side in 'LR'] for s in states])
    tracking=np.array([[np.linalg.norm(np.array(s['end_effector_actual'][side])[:3,3]-np.array(row['action']['end_effector_targets'][side])[:3,3]) for side in 'LR'] for row,s in zip(rows,states)])
    mask=np.array([s['geometry'] is not None for s in states]);valid=[s for s in states if s['geometry'] is not None];tv=t[mask]
    cuff=np.array([s['geometry']['axial_from_wrist_m'] for s in valid]);grips=np.array([[s['grasp_error_m'][side] for side in 'LR'] for s in valid])
    k=meta['config']['human']['stiffness'][0];label=f'K={k:g}, y={meta["offset_y_m"]:+.3f} m ({"success" if result["success"] else "failure"})'
    if meta['offset_y_m']==0.:
        for i,ax in enumerate(joint_axes.flat):ax.plot(t,q[:,i]-reference[i],label=f'K={k:g}')
    axes[0].plot(tv,cuff*100,label=label);axes[1].plot(t,q[:,1]-reference[1],label=label);axes[2].plot(t,tracking.max(1)*1000,label=label)
    np.savez_compressed(a.output/(run.name+'.npz'),time_s=t,human_q=q,robot_dq=dq,tracking_m=tracking,cuff_time_s=tv,cuff_axial_m=cuff,grasp_error_m=grips)
    summary.append({'episode':str(run),'success':result['success'],'termination_reason':result['termination_reason'],'sample_count':len(rows),'simulated_s':float(t[-1]),'clock_check':'passed',
        'K':k,'offset_y_m':meta['offset_y_m'],'maximum_tracking_error_m':tracking.max(0).tolist(),'maximum_precontrol_tracking_error_m':precontrol.max(0).tolist(),'maximum_grasp_error_m':grips.max(0).tolist() if len(grips) else None,
        'maximum_human_displacement_rad':np.abs(q-reference).max(0).tolist(),'maximum_robot_dq_including_jaws':float(np.max(np.abs(dq))),
        'final_geometry':valid[-1]['geometry'] if valid else None,'transitions':result['transitions'],
        'source_samples_sha256':hashlib.sha256((run/'samples.jsonl').read_bytes()).hexdigest()})
axes[0].set_ylabel('Cuff from wrist (cm)');axes[1].set_ylabel('Shoulder Y displacement (rad)');axes[2].set_ylabel('Max EE tracking error (mm)');axes[2].set_xlabel('Physical simulation time (s)')
for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(a.output/'task_responses.png',dpi=160);fig.savefig(a.output/'task_responses.pdf');(a.output/'analysis.json').write_text(json.dumps(summary,indent=2))
for name,ax in zip(('Shoulder X','Shoulder Y','Shoulder Z','Elbow Y'),joint_axes.flat):
    ax.set_title(name);ax.set_ylabel('q - q_ref (rad)');ax.set_xlabel('Simulation time (s)');ax.grid(alpha=.25);ax.legend()
joint_fig.tight_layout();joint_fig.savefig(a.output/'human_4dof_responses.png',dpi=160);joint_fig.savefig(a.output/'human_4dof_responses.pdf')
print(json.dumps(summary,indent=2))
