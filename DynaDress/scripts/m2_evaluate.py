"""Held-out whole parameter groups: one-step and open-loop prediction curves."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.models.world_model import CoupledWorldModel


def metrics(error,groups):
    result={}
    for name,unit,factor in [('human_q','rad',1),('human_dq','rad/s',1),('cuff_center','mm',1000),('progress','mm',1000),('cuff_points','mm',1000)]:
        x=error[...,slice(*groups[name])]
        result[f'{name}_rmse_{unit}']=float(np.sqrt(np.mean(x*x))*factor)
    x=error[...,slice(*groups['load'])].reshape(-1,2,6)
    result['load_force_rmse_N']=float(np.sqrt(np.mean(x[:,:,:3]**2)))
    result['load_torque_rmse_Nm']=float(np.sqrt(np.mean(x[:,:,3:]**2)))
    return result


def linear_rollout(history,horizon,fs):
    last=history[:,-1].copy();delta=last-history[:,-2]
    # A useful stronger baseline: observed motion extrapolation, load held.
    increment=np.zeros_like(delta);g=fs['groups'];dt=fs['dt_s']
    for group in ['ee_pose','cuff_center','cuff_points','progress']:
        increment[:,slice(*g[group])]=delta[:,slice(*g[group])]
    for pos,vel in [('human_q','human_dq'),('robot_q','robot_dq')]:
        increment[:,slice(*g[pos])]=dt*last[:,slice(*g[vel])]
    return np.stack([last+(h+1)*increment for h in range(horizon)],1)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'artifacts/m2/dataset');p.add_argument('--model',type=Path,default=ROOT/'artifacts/m2/model');p.add_argument('--output',type=Path,default=ROOT/'artifacts/m2/evaluation');p.add_argument('--device',default='cuda');p.add_argument('--split',choices=['validation','test'],default='test');a=p.parse_args()
    torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(a.model/'best.pt',map_location=a.device,weights_only=False)
    fs=checkpoint['feature_schema'];g=fs['groups'];h=fs['history_samples'];horizon=20
    model=CoupledWorldModel(checkpoint['normalizer'],fs,checkpoint['hidden'],checkpoint.get('transport_prior',False)).to(a.device)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    index=json.loads((a.dataset/'index.json').read_text());records=[r for r in index if r['split']==a.split]
    config=json.loads((a.model/'training_config.json').read_text())
    assert not set(config['train_groups']) & set(r['parameter_group'] for r in records)
    if a.split=='test':assert not set(config['validation_groups']) & set(r['parameter_group'] for r in records)
    pooled={name:[] for name in ['model','persistence','linear','mean_attributes']};contact_masks=[];per_episode=[];predictions={}
    with torch.no_grad():
        for record in records:
            d=dict(np.load(a.dataset/record['file']));obs=d['state'];act=d['action']
            anchors=np.arange(h,len(obs)-horizon,4)
            if len(anchors)==0:continue
            states=np.stack([obs[t-h+1:t+1] for t in anchors]);past=np.stack([act[t-h:t] for t in anchors]);actions=np.stack([act[t:t+horizon] for t in anchors]);truth=np.stack([obs[t+1:t+horizon+1] for t in anchors])
            cond=np.repeat(d['condition'][None],len(anchors),0)
            tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=a.device)
            pred=model.rollout(tensor(states),tensor(past),tensor(actions),tensor(cond)).cpu().numpy()
            mean=model.condition_mean[None].expand(len(anchors),-1)
            ablated=model.rollout(tensor(states),tensor(past),tensor(actions),mean).cpu().numpy()
            persist=np.repeat(states[:,-1:, :],horizon,1);linear=linear_rollout(states,horizon,fs)
            errors={name:val-truth for name,val in [('model',pred),('persistence',persist),('linear',linear),('mean_attributes',ablated)]}
            for name,e in errors.items():pooled[name].append(e)
            contact_masks.append(obs[anchors,g['progress'][0]]>-.05)
            per_episode.append(dict(episode_id=record['episode_id'],success=record['success'],windows=len(anchors),one_step=metrics(errors['model'][:,0],g),one_second=metrics(errors['model'][:,-1],g)))
            predictions[record['episode_id']]=(record,d,anchors,truth,pred,linear)
    summary=dict(evaluation_split=a.split,checkpoint_epoch=checkpoint['epoch'],checkpoint_validation_loss=checkpoint['validation_loss'],
        test_episodes=len(per_episode),test_parameter_groups=len(set(r['parameter_group'] for r in records)),
        horizons_s=[.05,.25,.5,1.],metrics={},metrics_near_wrist={},per_episode=per_episode,
        evaluation='Known held-out property combinations; recorded future actions supplied; no future measured state after history. Endpoints pooled over overlapping windows; not independent statistical trials.',
        baselines='persistence; measured-velocity/last-displacement extrapolation with held load; mean training attributes ablation',
        limitations='compact cuff state and estimated net wrist load; no claim of identification, calibrated contact force, policy control, or out-of-range generalization')
    pooled={name:np.concatenate(values) for name,values in pooled.items()}
    contact_mask=np.concatenate(contact_masks)
    summary['near_wrist_definition']='anchor cuff axial position > -50 mm from wrist; geometric subset, not a contact-force label'
    summary['test_windows']=len(contact_mask);summary['near_wrist_windows']=int(contact_mask.sum())
    for name,errors in pooled.items():
        summary['metrics'][name]={str((j+1)*fs['dt_s']):metrics(errors[:,j],g) for j in [0,4,9,19]}
        if contact_mask.any():summary['metrics_near_wrist'][name]={str((j+1)*fs['dt_s']):metrics(errors[contact_mask,j],g) for j in [0,4,9,19]}
    (a.output/'metrics.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(field,label) in zip(axes.flat,[('human_q_rmse_rad','Human q RMSE (rad)'),('cuff_center_rmse_mm','Cuff center RMSE (mm)'),('progress_rmse_mm','Progress RMSE (mm)'),('load_force_rmse_N','Estimated load RMSE (N)')]):
        for name in ['model','persistence','linear']:
            values=[metrics(pooled[name][:,j],g)[field] for j in range(horizon)]
            ax.plot(np.arange(1,horizon+1)*fs['dt_s'],values,label=name)
        ax.set(xlabel='Open-loop horizon (s)',ylabel=label);ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(a.output/'horizon_errors.png',dpi=170);fig.savefig(a.output/'horizon_errors.pdf');plt.close(fig)
    curve=json.loads((a.model/'learning_curve.json').read_text());fig,ax=plt.subplots(figsize=(8,4))
    for field in ['train_loss','validation_loss']:ax.plot([r['epoch'] for r in curve],[r[field] for r in curve],label=field)
    ax.set(xlabel='Epoch',ylabel='Weighted normalized MSE',yscale='log');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(a.output/'learning_curve.png',dpi=170);plt.close(fig)
    # A deterministic successful held-out probe, with a contact-rich 1 s forecast.
    candidates=[v for v in predictions.values() if v[0]['success'] and v[0]['profile']!='baseline'] or list(predictions.values())
    record,d,anchors,truth,pred,linear=candidates[0]
    desired_time=min(12.,float(d['time'][anchors[-1]]));wi=int(np.argmin(np.abs(d['time'][anchors]-desired_time)));t=anchors[wi]
    times=d['time'][t+1:t+horizon+1]
    dims=[(g['human_q'][0]+1,'Shoulder Y (rad)',1),(g['progress'][0],'Progress (mm)',1000),
          (g['load'][0]+2,'Left net wrist load Z (N)',1),(g['load'][0]+8,'Right net wrist load Z (N)',1)]
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,(dim,label,factor) in zip(axes.flat,dims):
        ax.plot(d['time'][max(0,t-12):t+1],d['state'][max(0,t-12):t+1,dim]*factor,color='gray',label='Observed history')
        ax.plot(times,truth[wi,:,dim]*factor,'k',label='PhysX future')
        ax.plot(times,pred[wi,:,dim]*factor,label='World model open loop')
        ax.plot(times,linear[wi,:,dim]*factor,'--',label='Linear baseline')
        ax.axvline(d['time'][t],color='gray',lw=.8);ax.set(xlabel='Simulation time (s)',ylabel=label);ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle(f'Held-out {record["episode_id"]}; K={record["K"]}, D={record["D"]}, friction={record["friction"]}')
    fig.tight_layout();fig.savefig(a.output/'prediction_example.png',dpi=170);fig.savefig(a.output/'prediction_example.pdf');plt.close(fig)
    np.savez_compressed(a.output/'prediction_example.npz',time=times,history=d['state'][t-h+1:t+1],true=truth[wi],predicted=pred[wi],linear=linear[wi],actions=d['action'][t:t+horizon],condition=d['condition'])
    (a.output/'prediction_example.json').write_text(json.dumps(dict(episode=record['episode_id'],anchor_time_s=float(d['time'][t]),features='dataset/features.json',horizon_s=1.),indent=2))
    print(json.dumps(summary['metrics'],indent=2))


if __name__=='__main__':main()
