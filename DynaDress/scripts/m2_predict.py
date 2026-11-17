"""Predict candidate actions from a compact episode history using a saved model."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.models.world_model import CoupledWorldModel
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--sequence',type=Path,required=True);p.add_argument('--anchor',type=int,default=160);p.add_argument('--horizon',type=int,default=20);p.add_argument('--actions',type=Path);p.add_argument('--condition',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
torch.set_num_threads(2)
c=torch.load(a.checkpoint,map_location=a.device,weights_only=False);fs=c['feature_schema'];h=fs['history_samples'];g=fs['groups'];d=np.load(a.sequence);t=a.anchor
if t<h or t>=len(d['state']):raise ValueError('Anchor lacks required past history')
actions=np.load(a.actions) if a.actions else d['action'][t:t+a.horizon]
if actions.shape!=(a.horizon,len(fs['action_names'])):raise ValueError('Provide horizon x action_dim candidate actions')
condition=np.array(json.loads(a.condition.read_text()),dtype=np.float32) if a.condition else d['condition']
if condition.shape!=(len(fs['condition_names']),):raise ValueError('Condition shape mismatch')
model=CoupledWorldModel(c['normalizer'],fs,c['hidden'],c.get('transport_prior',False)).to(a.device);model.load_state_dict(c['state_dict']);model.eval()
tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device=a.device)[None]
with torch.no_grad():pred=model.rollout(tensor(d['state'][t-h+1:t+1]),tensor(d['action'][t-h:t]),tensor(actions),tensor(condition)).cpu().numpy()[0]
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,predicted_state=pred,actions=actions,condition=condition,
    predicted_human_q=pred[:,slice(*g['human_q'])],predicted_human_dq=pred[:,slice(*g['human_dq'])],
    predicted_cuff_delta_m=pred[:,slice(*g['cuff_center'])]-d['state'][t,slice(*g['cuff_center'])],
    predicted_progress_delta_m=pred[:,g['progress'][0]]-d['state'][t,g['progress'][0]],
    predicted_estimated_load_N_Nm=pred[:,slice(*g['load'])])
print(json.dumps(dict(output=str(a.output),horizon_s=a.horizon*fs['dt_s'],action_source='candidate file' if a.actions else 'recorded future commands',future_state_input=False,force_source=fs['load_source'])))
