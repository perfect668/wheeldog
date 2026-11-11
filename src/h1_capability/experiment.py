"""M1-C experiment stages with complete statuses and frozen source provenance."""
from pathlib import Path
import argparse
import copy
from collections import Counter
import concurrent.futures
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import time
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from h1_kinematics.assets import Asset,unpack,dump,sha256
from h1_kinematics.collision import CollisionChecker
from .model import CapabilityModel,DIRECTIONS
from .force import ray_capacity
from .instantaneous import cooperative_velocity
from .pairing import match_pairs
from .reachability import search


def frozen_inventory():
    entries={}
    for name in ['results/diagnostics/artifact_sha256.json','results/m1/artifact_sha256.json']:
        contents=json.loads(Path(name).read_text())
        for p,h in contents.items():
            actual=sha256(p)
            if actual!=h:raise ValueError(f'Frozen artifact changed: {p}')
        entries[name]=dict(count=len(contents),all_match=True,inventory_sha256=sha256(name))
    return entries


def metadata_record(config):
    git=lambda *args:subprocess.check_output(['git',*args],text=True).strip()
    return dict(source_commit=git('rev-parse','HEAD'),tracked_status=git('status','--porcelain','--untracked-files=no'),
        seed=config['seed'],python=platform.python_version(),platform=platform.platform(),
        versions={n:metadata.version(n) for n in ['numpy','scipy','mujoco','trimesh','python-fcl','matplotlib','pytest','pyyaml']},
        input_hashes={p:sha256(p) for p in [config['archive'],config['candidates'],config['world_targets']]},
        frozen_integrity=frozen_inventory(),gravity_model='none: pure contact torque budget relaxation',
        total_mass_usage='190kg provenance only, not distributed or used in stability',
        provenance='URDF geometry, research-lead arm overrides; motion dynamics assumed, no hardware validation')


def setup(config):
    asset=Asset(unpack(config['archive'],config['asset_directory']))
    return asset,CapabilityModel(asset,config['model']),CollisionChecker(asset)


def force_stage(config,out):
    started=time.perf_counter();asset,model,checker=setup(config)
    rows=json.loads(Path(config['candidates']).read_text());reference=json.loads(Path(config['world_targets']).read_text())
    targets=[np.array(p)@h for p,h in zip(reference['world_targets'],model.handles)]
    task=np.r_[config['force']['task_left_n'],config['force']['task_right_n']];records=[]
    for row in rows:
        record=dict(id=row['id'],q_rad=row.get('q_rad'),base_xy_yaw=row['base_xy_yaw'],status='REJECTED',reasons=[],source_status=row['status'])
        if row['status']!='ACCEPTED_CONDITIONAL':record['reasons'].append('source_m1b_rejected');records.append(record);continue
        q=np.array(row['q_rad']);base=np.array(row['base_xy_yaw']);poses=model.poses(q,base)
        errors=[dict(position_m=float(np.linalg.norm(t[:3,3]-p[:3,3])),rotation_rad=float(Rotation.from_matrix(t[:3,:3]@p[:3,:3].T).magnitude())) for t,p in zip(poses,targets)]
        record['handle_world_poses']=[p.tolist() for p in poses];record['target_errors']=errors
        if max(e['position_m'] for e in errors)>config['motion']['pose_position_tolerance_m'] or max(e['rotation_rad'] for e in errors)>config['motion']['pose_rotation_tolerance_rad']:record['reasons'].append('pose_target')
        if np.any(q<model.robot.lower-1e-9) or np.any(q>model.robot.upper+1e-9):record['reasons'].append('joint_limit_overlay')
        record['collision']=checker.check(model.robot.fk(q,base),config['motion']['self_clearance_m'])
        if not record['collision']['passed']:record['reasons'].append('collision')
        contact=model.contact_map(q,base);record['contact_map_nm_per_n']=contact.tolist()
        record['task_compensating_torque_nm']=(contact@task).tolist()
        record['shared_torso_torque_nm']=(contact@task)[model.torso_indices].tolist()
        record['shared_torso_joint_names']=model.torso_names
        record['arm_position_margin_rad']=[float(min(q[i]-model.robot.lower[i],model.robot.upper[i]-q[i])) for i in model.active]
        record['force']={};record['instantaneous']={};vector=[]
        for budget,scale in [('nominal',1.),('urdf',1.),('nominal',.5)]:
            tag=budget if scale==1 else 'nominal_half';record['force'][tag]={}
            for kind,f0 in [('unloaded',None),('residual',task)]:
                record['force'][tag][kind]={}
                for name,d in DIRECTIONS.items():
                    result=ray_capacity(contact,model.budgets(budget,scale),d,f0)
                    if result['status']=='OPTIMAL':result['limiting_joint_names']=[model.robot.names[i] for i in result['limiting_dofs']]
                    record['force'][tag][kind][name]=result
                    if budget=='nominal' and scale==1:vector.append(result.get('capacity_n'))
        if any(x is None for x in vector):record['reasons'].append('force_not_finite_optimal')
        else:record['force_match_vector_n']=vector
        for mobile in [False,True]:
            record['instantaneous']['mobile' if mobile else 'fixed']={name:cooperative_velocity(model,q,base,d,config['profiles']['nominal'],mobile) for name,d in DIRECTIONS.items()}
        record['status']='REJECTED' if record['reasons'] else 'ACCEPTED'
        records.append(record)
    dump(out/'force_and_instantaneous.json',records)
    pairs=match_pairs(records,config['force']['match_relative_tolerance'],config['force']['match_floor_n'])
    dump(out/'all_force_pairs.json',pairs)
    mapping=dict(arm_joint_names_in_chain_order=model.arm_names,arm_indices=model.arm_indices,
                 shared_torso_joint_names=model.torso_names,shared_torso_indices=model.torso_indices,
                 active_motion_joint_names=[model.robot.names[i] for i in model.active],
                 unconstrained_shared_torque=model.config['torso_torque_limits_nm'] is None,
                 raw_urdf_unchanged=True,force_sign='robot ON human; robot reaction -f; balancing actuator +J^T f')
    dump(out/'model_mapping.json',mapping)
    print('force stage',Counter(r['status'] for r in records),'matches',sum(p['status']=='MATCHED' for p in pairs),'seconds',time.perf_counter()-started,flush=True)
    return records


WORKER=None

def init_worker(config):
    global WORKER
    WORKER=(config,*setup(config))


def compute_case(case):
    config,asset,model0,checker=WORKER
    effective=copy.deepcopy(config['model'])
    if 'joint7_deg' in case:effective['joint_ranges_deg'][6]=[-case['joint7_deg'],case['joint7_deg']]
    model=model0 if effective==config['model'] else CapabilityModel(asset,effective)
    options=copy.deepcopy(config['motion']);options.update(case.get('motion_overrides',{}))
    profile=copy.deepcopy(config['profiles'][case.get('profile','nominal')]);profile.update(case.get('profile_overrides',{}))
    q=np.array(case['q_rad']);base=np.array(case['base_xy_yaw'])
    task=np.r_[config['force']['task_left_n'],config['force']['task_right_n']]
    if np.any(q<model.robot.lower-1e-9) or np.any(q>model.robot.upper+1e-9):
        result=dict(status='INITIAL_CONFIG_REJECTED',achieved_displacement_m=None,attempts=[],reason='position_limit_under_variant')
    else:
        result=search(model,checker,q,base,DIRECTIONS[case['direction']],case['horizon_s'],profile,case['mobile'],options,task,case['path_file'])
    result.update({k:v for k,v in case.items() if k not in ['q_rad','base_xy_yaw']})
    return result


def execute(config,cases,out,workers):
    case_dir=out/'cases';case_dir.mkdir(exist_ok=True);(out/'paths').mkdir(exist_ok=True)
    pending=[case for case in cases if not (case_dir/(case['case_id']+'.json')).exists()]
    if workers==1:
        init_worker(config)
        for i,case in enumerate(pending):
            r=compute_case(case);dump(case_dir/(r['case_id']+'.json'),r)
            if i%20==0:print('cases',i+1,'/',len(pending),r['case_id'],flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(config,)) as pool:
            futures={pool.submit(compute_case,case):case for case in pending}
            for i,f in enumerate(concurrent.futures.as_completed(futures)):
                r=f.result();dump(case_dir/(r['case_id']+'.json'),r)
                if i%40==0:print('cases',i+1,'/',len(pending),r['case_id'],flush=True)
    results=[json.loads((case_dir/(case['case_id']+'.json')).read_text()) for case in cases]
    return results


def make_cases(records,out,horizons,tag='main',profiles=('nominal',),joint7=None,profile_overrides=None,motion_overrides=None):
    cases=[]
    for r in records:
        if r['status']!='ACCEPTED':continue
        for profile in profiles:
            for h in horizons:
                for mobile in [False,True]:
                    for name in DIRECTIONS:
                        case_id=f'{tag}_{r["id"]}_{profile}_{h:g}_{"mobile" if mobile else "fixed"}_{name}'
                        case=dict(case_id=case_id,candidate=r['id'],q_rad=r['q_rad'],base_xy_yaw=r['base_xy_yaw'],
                                  direction=name,horizon_s=h,mobile=mobile,profile=profile,
                                  path_file=str(out/'paths'/(case_id+'.npz')),experiment=tag)
                        if joint7 is not None:case['joint7_deg']=joint7
                        if profile_overrides:case['profile_overrides']=profile_overrides
                        if motion_overrides:case['motion_overrides']=motion_overrides
                        cases.append(case)
    return cases


def include_fixed_option(results,baseline=()):
    """A mobile base can remain stationary: union with verified fixed-base path.

    Keep the original local search outcome; the union is a feasible lower bound,
    not a monotonicity assertion about the incomplete local solver.
    """
    lookup={(r['experiment'],r['candidate'],r['direction'],r['horizon_s']):r for r in results if not r['mobile']}
    fallback={(r['candidate'],r['direction'],r['horizon_s']):r for r in baseline if not r['mobile']}
    output=[]
    for row in results:
        r=copy.deepcopy(row)
        if r['mobile']:
            fixed=lookup.get((r['experiment'],r['candidate'],r['direction'],r['horizon_s']))
            if fixed is None and r['experiment'].startswith('base_acc_'):
                fixed=fallback.get((r['candidate'],r['direction'],r['horizon_s']))
            if fixed and (fixed.get('achieved_displacement_m') or 0)>(r.get('achieved_displacement_m') or 0):
                r['local_mobile_search']=copy.deepcopy(row)
                for key in ['status','achieved_displacement_m','capped','local_search_upper_m','best','path_file']:
                    r[key]=copy.deepcopy(fixed.get(key))
                r['selected_path_origin']='fixed_base_subset'
            else:r['selected_path_origin']='mobile_local_search'
        output.append(r)
    return output


def flatten(results,path):
    with path.open('w',newline='') as f:
        fields=['case_id','experiment','candidate','direction','horizon_s','mobile','status','achieved_displacement_m','capped','elapsed_s']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)


def run(config_path,output,stage,workers):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    c=yaml.safe_load(Path(config_path).read_text())
    saved=out/'config.yaml'
    if saved.exists() and yaml.safe_load(saved.read_text())!=c:raise ValueError('Output has different configuration')
    if not saved.exists():
        saved.write_text(yaml.safe_dump(c,sort_keys=False));dump(out/'metadata.json',metadata_record(c))
    if stage=='force':force_stage(c,out);return
    records=json.loads((out/'force_and_instantaneous.json').read_text())
    if stage=='reach':
        cases=make_cases(records,out,c['motion']['horizons_s']);results=execute(c,cases,out,workers)
        results=include_fixed_option(results)
        dump(out/'reachability.json',results);flatten(results,out/'reachability.csv');return
    if stage=='sensitivity':
        pairs=json.loads((out/'selected_pairs.json').read_text())
        ids=sorted(set(x for p in pairs for x in [p['a'],p['b']]))
        subset=[r for r in records if r['id'] in ids];h=[c['sensitivity']['horizon_s']];cases=[]
        for name in ['low','high']:cases+=make_cases(subset,out,h,tag='profile_'+name,profiles=(name,))
        cases+=make_cases(subset,out,h,tag='joint7_60',joint7=60)
        for value in c['sensitivity']['acceleration_rad_s2']:
            if value!=c['profiles']['nominal']['joint_acceleration_rad_s2']:cases+=make_cases(subset,out,h,tag=f'joint_acc_{value:g}',profile_overrides={'joint_acceleration_rad_s2':value})
        for value in c['sensitivity']['delays_s']:
            if value!=c['profiles']['nominal']['delay_s']:cases+=make_cases(subset,out,h,tag=f'delay_{value:g}',profile_overrides={'delay_s':value})
        for value in c['sensitivity']['base_accelerations_m_s2']:
            if value!=c['profiles']['nominal']['base_acceleration_m_s2']:
                extra=make_cases(subset,out,h,tag=f'base_acc_{value:g}',profile_overrides={'base_acceleration_m_s2':value});cases+=[x for x in extra if x['mobile']]
        cases+=make_cases(subset,out,h,tag='dt_0.01',motion_overrides={'dt_s':.01})
        cases+=make_cases(subset,out,h,tag='uniform_policy',motion_overrides={'inverse_policies':['uniform']})
        results=execute(c,cases,out,workers)
        results=include_fixed_option(results,json.loads((out/'reachability.json').read_text()))
        dump(out/'sensitivity.json',results);flatten(results,out/'sensitivity.csv');return
    raise ValueError(stage)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/m1_capability.yaml');p.add_argument('--output',required=True)
    p.add_argument('--stage',choices=['force','reach','sensitivity'],required=True);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();run(a.config,a.output,a.stage,a.workers)
