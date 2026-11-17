"""Post-test diagnostics; no fitting, filtering, exclusions or score changes."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.models.world_model import CoupledWorldModel
base=ROOT/'artifacts/m2';out=base/'analysis';out.mkdir(parents=True,exist_ok=True)
index=json.loads((base/'dataset/index.json').read_text());fs=json.loads((base/'dataset/features.json').read_text());g=fs['groups'];extrema=[]
for item in index:
    data=np.load(base/'dataset'/item['file']);load=data['state'][:,slice(*g['load'])].reshape(-1,2,6);force=np.linalg.norm(load[:,:,:3],axis=-1)
    extrema.append(dict(episode=item['episode_id'],split=item['split'],max_wrist_force_N=float(force.max()),p95_wrist_force_N=float(np.percentile(force,95)),maximum_human_dq_rad_s=float(np.abs(data['state'][:,slice(*g['human_dq'])]).max())))
extrema.sort(key=lambda r:r['max_wrist_force_N'],reverse=True);(out/'load_extrema.json').write_text(json.dumps(extrema,indent=2))
eid=extrema[0]['episode'];run=base/'episodes'/eid;rows=[json.loads(r) for r in (run/'samples.jsonl').read_text().splitlines()]
force=np.array([[r['estimated_load']['value']['external_world_N_Nm'][s][:3] for s in 'LR'] for r in rows]);i=int(np.argmax(np.linalg.norm(force,axis=-1).max(1)))
selected=rows[max(0,i-10):i+11];time=np.array([r['simulation_time_s'] for r in selected]);est=np.array([[r['estimated_load']['value']['external_world_N_Nm'][s] for s in 'LR'] for r in selected]);raw=np.array([[r['solver_state']['value']['wrist_reaction']['raw_child_frame_N_Nm'][s] for s in 'LR'] for r in selected])
(out/'load_peak_source_excerpt.json').write_text(json.dumps(dict(episode=eid,center_sample_index=i,samples=selected),indent=2))
fig,axes=plt.subplots(2,2,figsize=(11,7))
for s,side in enumerate('LR'):
    axes[0,0].plot(time,np.linalg.norm(raw[:,s,:3],axis=-1),label=side)
    axes[0,1].plot(time,np.linalg.norm(est[:,s,:3],axis=-1),label=side)
    axes[1,1].plot(time,[r['solver_state']['value']['end_effector_tracking_error_m'][side]*1000 for r in selected],label=side)
for j,name in enumerate(['shoulder X','shoulder Y','shoulder Z','elbow']):axes[1,0].plot(time,[r['solver_state']['value']['human_q_rad'][j] for r in selected],label=name)
for ax,label in zip(axes.flat,['Raw incoming wrist force norm (N)','Estimated net external force norm (N)','Human q (rad)','EE tracking error (mm)']):ax.set(xlabel='Simulation time (s)',ylabel=label);ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle(eid+' — native solver spike retained');fig.tight_layout();fig.savefig(out/'load_peak.png',dpi=170);plt.close(fig)
metrics=json.loads((base/'evaluation/metrics.json').read_text());worst=max(metrics['per_episode'],key=lambda r:r['one_second']['human_q_rmse_rad']);eid=worst['episode_id'];data=np.load(base/'dataset'/f'{eid}.npz')
torch.set_num_threads(2);c=torch.load(base/'model/best.pt',map_location='cpu',weights_only=False);model=CoupledWorldModel(c['normalizer'],c['feature_schema'],c['hidden'],c.get('transport_prior',False));model.load_state_dict(c['state_dict']);model.eval();h=fs['history_samples'];horizon=20
anchors=np.arange(h,len(data['state'])-horizon,4);tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
states=np.stack([data['state'][t-h+1:t+1] for t in anchors]);past=np.stack([data['action'][t-h:t] for t in anchors]);actions=np.stack([data['action'][t:t+horizon] for t in anchors]);truth=np.stack([data['state'][t+1:t+horizon+1] for t in anchors])
with torch.no_grad():pred=model.rollout(tensor(states),tensor(past),tensor(actions),tensor(np.repeat(data['condition'][None],len(anchors),0))).numpy()
errors=np.mean((pred[:,-1,slice(*g['human_q'])]-truth[:,-1,slice(*g['human_q'])])**2,axis=-1);wi=int(np.argmax(errors));t=anchors[wi];future_time=data['time'][t+1:t+horizon+1];fig,axes=plt.subplots(2,2,figsize=(11,7))
for j,(ax,name) in enumerate(zip(axes.flat,['shoulder X','shoulder Y','shoulder Z','elbow'])):
    dim=g['human_q'][0]+j
    ax.plot(data['time'][t-h+1:t+1],data['state'][t-h+1:t+1,dim],color='gray',label='Observed history')
    ax.plot(future_time,truth[wi,:,dim],'k',label='PhysX future');ax.plot(future_time,pred[wi,:,dim],label='World model')
    ax.plot(future_time,np.repeat(data['state'][t,dim],horizon),'--',label='Persistence')
    ax.set(xlabel='Simulation time (s)',ylabel=name+' (rad)');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle(eid+' — worst 1 s human-q window in worst test episode');fig.tight_layout();fig.savefig(out/'worst_human_prediction.png',dpi=170);plt.close(fig)
(out/'post_test_diagnostics.json').write_text(json.dumps(dict(load_peak_episode=extrema[0]['episode'],load_peak_simulation_s=rows[i]['simulation_time_s'],worst_human_episode=eid,worst_human_anchor_s=float(data['time'][t]),selection='post-test extrema for diagnosis only; model, data and primary metrics unchanged'),indent=2))
print(json.dumps(dict(load_peak=extrema[0],worst_human_episode=worst)))
