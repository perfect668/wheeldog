"""Validate real logs, freeze group-disjoint splits and export compact sequences."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.models.features import extract,attributes,schema
p=argparse.ArgumentParser();p.add_argument('--episodes',type=Path,default=ROOT/'artifacts/m2/episodes');p.add_argument('--output',type=Path,default=ROOT/'artifacts/m2/dataset');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
plan=json.loads((ROOT/'dynadress/configs/m2_collection/plan.json').read_text())['episodes']
groups=sorted(set(r['parameter_group'] for r in plan));np.random.default_rng(1702).shuffle(groups)
split={g:('train' if i<35 else 'validation' if i<42 else 'test') for i,g in enumerate(groups)}
(a.output/'split_groups.json').write_text(json.dumps(split,indent=2))
index=[];missing=[];feature_schema=None
for item in plan:
    eid=item['episode_id'];run=a.episodes/eid
    if not (run/'metadata.json').exists() or not json.loads((run/'metadata.json').read_text()).get('complete'):
        missing.append(eid);continue
    cfg=json.loads((run/'config.json').read_text());meta=json.loads((run/'metadata.json').read_text());result=json.loads((run/'result.json').read_text())
    rows=[json.loads(s) for s in (run/'samples.jsonl').read_text().splitlines()]
    times=np.array([r['simulation_time_s'] for r in rows]);steps=np.array([r['physics_step'] for r in rows]);wall=np.array([r['sample_wall_time_ns'] for r in rows])
    np.testing.assert_allclose(np.diff(times),np.diff(steps)/240,atol=2e-6)
    assert np.all(np.diff(steps)==12) and np.all(np.diff(wall)>0)
    assert all(np.array_equal(r['privileged']['q_ref'],cfg['human']['reference']) for r in rows)
    assert all(r['solver_contact']=={'available':False,'value':None} and r['measured_wrench']=={'available':False,'value':None} for r in rows)
    assert np.array_equal(cfg['human']['stiffness'],[item['K']]*4) and np.array_equal(cfg['human']['damping'],[item['D']]*4)
    assert cfg['garment']['friction']==item['friction']
    assert cfg['parameter_group']==item['parameter_group']
    if (run/'material_readback.json').exists():
        assert all(abs(float(v)-item['friction'])<1e-6 for v in json.loads((run/'material_readback.json').read_text()).values() if v is not None)
    names=rows[0]['solver_state']['value']['robot_joint_names'];fs=schema(names)
    if feature_schema is None:feature_schema=fs
    assert fs==feature_schema
    valid=[i for i,r in enumerate(rows) if extract(r) is not None]
    if valid:assert valid==list(range(valid[0],valid[-1]+1))
    features=[extract(rows[i]) for i in valid]
    if len(features)<12:
        missing.append(eid+':insufficient_valid_states');continue
    obs=np.stack([f[0] for f in features]);act=np.stack([f[1] for f in features])
    assert obs.shape[1]==len(fs['state_names']) and act.shape[1]==len(fs['action_names'])
    target=a.output/f'{eid}.npz'
    np.savez_compressed(target,state=obs,action=act,condition=attributes(cfg),time=times[valid],physics_step=steps[valid])
    load_start=fs['groups']['load'][0];pre=np.array([r['estimated_load']['value']['external_world_N_Nm'][s] for r in rows if .2<r['simulation_time_s']<.7 for s in 'LR'])
    index.append(dict(**item,split=split[item['parameter_group']],samples=len(rows),valid_states=len(obs),file=target.name,success=result['success'],termination_reason=result['termination_reason'],
                      simulated_seconds=result['simulation_duration_s'],primitive_counts=dict(collections.Counter(r['action']['primitive'] for r in rows)),source_sha256=hashlib.sha256((run/'samples.jsonl').read_bytes()).hexdigest(),
                      max_no_cloth_residual_force_N=float(np.max(np.linalg.norm(pre[:,:3],axis=1))),
                      max_no_cloth_residual_torque_Nm=float(np.max(np.linalg.norm(pre[:,3:],axis=1))),
                      code_sha256=meta['code_sha256']))
assert feature_schema is not None,'No complete valid episodes'
assert len(set(r['episode_id'] for r in index))==len(index)
sets={s:{r['parameter_group'] for r in index if r['split']==s} for s in ['train','validation','test']}
assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test'])
(a.output/'features.json').write_text(json.dumps(feature_schema,indent=2));(a.output/'index.json').write_text(json.dumps(index,indent=2))
(a.output/'validation.json').write_text(json.dumps(dict(complete_episodes=len(index),missing=missing,successes=sum(r['success'] for r in index),
    splits={s:dict(episodes=sum(r['split']==s for r in index),parameter_groups=sum(v==s for v in split.values())) for s in ['train','validation','test']},
    checks=['240Hz physical clock / 20Hz control','fixed reference','complete episode and parameter group split','finite features; no missing load imputation','unavailable contacts and physical sensor preserved','parameter plan consistency'],
    legacy_m11='six earlier episodes retained separately; no new load measurement, excluded from supervised load targets'),indent=2))
print(json.dumps(dict(episodes=len(index),missing=len(missing),states=sum(r['valid_states'] for r in index))))
