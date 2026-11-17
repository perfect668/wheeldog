"""Audit recorded FEM nodes, clock, containment history and command provenance."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from dynadress.garments.cuff import contains_section
p=argparse.ArgumentParser();p.add_argument('episode',type=Path);a=p.parse_args();run=a.episode
rows=[json.loads(line) for line in (run/'samples.jsonl').read_text().splitlines()]
meta=json.loads((run/'metadata.json').read_text());result=json.loads((run/'result.json').read_text())
state=[r['solver_state']['value'] for r in rows];time=np.array([r['simulation_time_s'] for r in rows]);step=np.array([r['physics_step'] for r in rows]);cfg=meta['config']
assert np.all(np.diff(step)>0);np.testing.assert_allclose(np.diff(time),np.diff(step)/240,atol=2e-6)
for r in rows:
    assert r['privileged']['q_ref']==cfg['human']['reference']
    for key in ('solver_contact','estimated_load','measured_wrench'):assert r[key]=={'available':False,'value':None}
final=np.load(run/'cloth_final.npz');pts=final['positions'];cuff=final['cuff'];s=state[-1]
np.testing.assert_allclose(pts[cuff],s['cuff_keypoints_m'],atol=1e-6)
criteria=[]
for row,s in zip(rows[-10:],state[-10:]):
    ring=np.array(s['cuff_keypoints_m']);wrist=np.array(s['human_wrist_m']);elbow=np.array(s['human_elbow_m']);axis=elbow-wrist;axis/=np.linalg.norm(axis)
    test=contains_section(ring,wrist,.035,axis);progress=float((ring.mean(0)-wrist)@axis)
    # Recompute tracking from each actual pose, not the separately logged error.
    tracking={side:float(np.linalg.norm(np.array(s['end_effector_actual'][side])[:3,3]-np.array(row['action']['end_effector_targets'][side])[:3,3])) for side in 'LR'}
    criteria.append(bool(test['inside'] and test['plane_distance_m']<-.005 and progress>=.26*cfg['target_forearm_fraction']-.004 and max(s['grasp_error_m'].values())<.01 and max(tracking.values())<.008 and s['hand_entered'] and s['wrist_entered']))
s=state[-1];elbow=np.array(s['human_elbow_m']);wrist=np.array(s['human_wrist_m']);axis=elbow-wrist;axis/=np.linalg.norm(axis)
penetration={}
for name,lo,hi in [('hand',-.35,-.26),('forearm',-.225,-.035)]:
    aa=elbow+lo*axis;ab=(hi-lo)*axis;t=np.clip((pts-aa)@ab/(ab@ab),0,1)
    sd=np.linalg.norm(pts-aa-t[:,None]*ab,axis=1)-.035
    penetration[name]={'minimum_node_signed_distance_m':float(sd.min()),'nodes_inside_capsule':int(sum(sd<0)),'nodes_deeper_than_3mm':int(sum(sd<-.003)),'scope':'final nodes only; excludes triangle interiors and continuous-time penetration'}
valid=bool(result['success'] and result['termination_reason']=='geometric_completion' and all(criteria))
audit={'verified_recorded_completion':valid,'clock':'passed','fixed_reference':'passed','unavailable_force_provenance':'passed','cloth_snapshot_matches_record':'passed',
       'last_ten_criteria':criteria,'last_ten_sample_span_s':float(time[-1]-time[-10]),'final_capsule_node_check':penetration,
       'endpoint_tracking_source':'FK of measured solver joint angles; fixed robot base','not_evaluated':'solver contact force, full mesh continuous collision certification, frictional grasp strength'}
(run/'validation.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
if result['success'] and not valid:raise SystemExit('Success flag did not survive record audit')
