"""Saved-record replay: numerical consistency, not physical model validation."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from h1_kinematics.assets import Asset,unpack,dump
from h1_kinematics.model import Robot,TIPS
from h1_kinematics.collision import CollisionChecker
from h1_kinematics.experiment import evaluate

p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();out=Path(a.output)
read=lambda n:json.loads((out/n).read_text())
c=yaml.safe_load((out/'config.yaml').read_text());r=Robot(Asset(unpack(c['archive'],c['asset_directory'])));k=CollisionChecker(r.asset)
targets=np.array(read('targets.json')['world_targets']);rows=[]
for dataset in ['candidates.json','random_validation.json']:
    for row in read(dataset):
        if row['status']=='NUMERICAL_ERROR':
            rows.append(dict(id=row['id'],matched=None,recorded_numerical_error=True));continue
        current=evaluate(r,k,np.array(row['q_rad']),np.array(row['base_xy_yaw']),c,
                         targets if dataset=='candidates.json' else None)
        rows.append(dict(id=row['id'],matched=current['status']==row['status'] and current['reasons']==row['reasons']))
z=np.load(out/'configurations.npz',allow_pickle=False);candidates=read('candidates.json')
assert np.array_equal(z['joint_names'],r.names)
assert np.array_equal(z['all_q_rad'],np.array([x['q_rad'] for x in candidates]))
assert np.array_equal(z['world_targets'],targets)
accepted=[np.array(x['q_rad'])[r.active] for x in candidates if x['status']=='ACCEPTED_CONDITIONAL']
distance=min(np.linalg.norm(a-b) for i,a in enumerate(accepted) for b in accepted[i+1:]) if len(accepted)>1 else None
random_rows=[x for x in read('random_validation.json') if x['status']=='ACCEPTED_CONDITIONAL']
full=np.array([r.jacobians(np.array(x['q_rad']),x['base_xy_yaw']) for x in random_rows])
left=[r.dof[n] for n in r.names if '08L' in n];right=[r.dof[n] for n in r.names if '08R' in n]
np.savez_compressed(out/'jacobians.npz',ids=np.array([x['id'] for x in random_rows]),
    full_world_linear_angular=full,left_arm_6x7=full[:,0,:,:][:,:,left],
    right_arm_6x7=full[:,1,:,:][:,:,right],joint_names=np.array(r.names),
    left_joint_names=np.array([r.names[k] for k in left]),right_joint_names=np.array([r.names[k] for k in right]))
report=dict(records=rows,status_mismatches=sum(row.get('matched') is False for row in rows),
            checked=len(rows),npz_json_match=True,min_accepted_joint_l2_separation_rad=distance,
            qualification='same model replay only; not hardware/solid occupancy validation')
dump(out/'replay_validation.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
if report['status_mismatches']: raise SystemExit(1)
