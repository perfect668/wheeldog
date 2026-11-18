"""Audit hold saturation/joint-limit regime on all frozen original M2 states."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.models.impedance import gravity_compensation,JOINT_LOWER,JOINT_UPPER,TORQUE_LIMIT
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'artifacts/m2/dataset');p.add_argument('--output',type=Path,default=ROOT/'artifacts/m21/dynamics/data_regime_audit.json');a=p.parse_args();torch.set_num_threads(1)
index=json.loads((a.dataset/'index.json').read_text());report={'scope':'All 20 Hz human states retained in original M2 compact dataset; does not observe intermediate 240 Hz extrema','source_index_sha256':hashlib.sha256((a.dataset/'index.json').read_bytes()).hexdigest(),'splits':{}}
for split in ['train','validation','test','all']:
 qs=[];vs=[];restores=[];holds=[];episodes=[]
 for item in index:
  if split!='all' and item['split']!=split:continue
  z=np.load(a.dataset/item['file']);q=z['state'][:,:4].astype('float64');v=z['state'][:,4:8].astype('float64');c=z['condition'];restoring=c[:4]*(c[9:13]-q);g=gravity_compensation(torch.tensor(q)).numpy();hold=g+restoring
  qs.append(q);vs.append(v);restores.append(restoring);holds.append(hold);episodes.append(item['episode_id'])
 q=np.concatenate(qs);v=np.concatenate(vs);hold=np.concatenate(holds);restoring=np.concatenate(restores);margin=np.minimum(q-np.array(JOINT_LOWER),np.array(JOINT_UPPER)-q)
 report['splits'][split]={'episodes':len(episodes),'states':len(q),'q_min_rad':q.min(0).tolist(),'q_max_rad':q.max(0).tolist(),'dq_max_abs_rad_s':abs(v).max(0).tolist(),'joint_limit_min_margin_rad':margin.min(0).tolist(),'states_within_001rad_of_limit':(margin<.01).sum(0).tolist(),'outside_limit_states':(margin< -1e-6).sum(0).tolist(),'explicit_raw_hold_max_abs_Nm':abs(hold).max(0).tolist(),'explicit_raw_hold_clip_states':(abs(hold)>TORQUE_LIMIT).sum(0).tolist(),'K_restoring_max_abs_Nm':abs(restoring).max(0).tolist()}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
