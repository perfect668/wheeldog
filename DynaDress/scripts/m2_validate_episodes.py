"""Audit every recorded success independently of its saved success flag."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.garments.cuff import contains_section
p=argparse.ArgumentParser();p.add_argument('--episodes',type=Path,default=ROOT/'artifacts/m2/episodes');a=p.parse_args();audits=[]
for run in sorted(a.episodes.iterdir()):
    if not (run/'metadata.json').exists() or not json.loads((run/'metadata.json').read_text()).get('complete'):continue
    rows=[json.loads(s) for s in (run/'samples.jsonl').read_text().splitlines()];cfg=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
    criteria=[]
    for row in rows[-10:]:
        s=row['solver_state']['value'];ring=np.asarray(s['cuff_keypoints_m']);wrist=np.asarray(s['human_wrist_m']);axis=np.asarray(s['human_elbow_m'])-wrist;axis/=np.linalg.norm(axis)
        test=contains_section(ring,wrist,.035,axis);progress=float((ring.mean(0)-wrist)@axis)
        tracking=max(np.linalg.norm(np.asarray(s['end_effector_actual'][side])[:3,3]-np.asarray(row['action']['end_effector_targets'][side])[:3,3]) for side in 'LR')
        criteria.append(bool(test['inside'] and test['plane_distance_m']<-.005 and progress>=.26*cfg['target_forearm_fraction']-.004 and max(s['grasp_error_m'].values())<.01 and tracking<.008 and s['hand_entered'] and s['wrist_entered']))
    if result['success']:
        assert len(criteria)==10 and all(criteria),f'{run}: completion rejected'
        final=np.load(run/'cloth_final.npz');np.testing.assert_allclose(final['positions'][final['cuff']],rows[-1]['solver_state']['value']['cuff_keypoints_m'],atol=1e-6)
    # Explicitly account for the final non-sampled physics interval on a timeout.
    snapshot_gap=result['simulation_duration_s']-(rows[-1]['simulation_time_s']-rows[0]['simulation_time_s'])
    max_step=max(np.linalg.norm(r['action']['end_effector_delta_world_m'][side]) for r in rows for side in 'LR')
    assert max_step<=cfg['end_effector_speed_m_s']*.05+1e-6
    audit=dict(episode=run.name,success=result['success'],termination_reason=result['termination_reason'],verified_success=bool(result['success'] and all(criteria)),
               last_ten_criteria=criteria,last_ten_span_s=rows[-1]['simulation_time_s']-rows[-10]['simulation_time_s'],
               max_command_delta_m=float(max_step),recorded_sample_count=len(rows),clock_checked_by='m2_dataset.py',
               final_snapshot_after_last_sample_s=float(snapshot_gap),
               snapshot_scope='success snapshots match terminal sampled cuff; timeout can end after last sample; no continuous penetration certification')
    (run/'validation_m2.json').write_text(json.dumps(audit,indent=2));audits.append(audit)
out=ROOT/'artifacts/m2/analysis';out.mkdir(parents=True,exist_ok=True);(out/'episode_audit.json').write_text(json.dumps(audits,indent=2));print(json.dumps(dict(audited=len(audits),verified_successes=sum(r['verified_success'] for r in audits),failures=sum(not r['success'] for r in audits))))
