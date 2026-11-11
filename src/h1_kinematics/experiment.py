"""Reproducible static M1 validation. Every attempted sample is retained."""
from pathlib import Path
import csv
import json
import platform
import subprocess
import time
import importlib.metadata as metadata
import numpy as np
import yaml
from .assets import Asset, unpack, dump, sha256
from .model import Robot, TIPS
from .collision import CollisionChecker


def evaluate(robot,checker,q,base,config,targets=None):
    reasons=[]
    if not np.isfinite(q).all() or not np.isfinite(base).all():
        return dict(status='NUMERICAL_ERROR',reasons=['nonfinite_configuration'])
    violations=[n for n,k in robot.index.items() if q[k]<robot.lower[k]-1e-10 or q[k]>robot.upper[k]+1e-10]
    if violations: reasons.append('joint_limit')
    e=robot.errors(q,base,targets) if targets is not None else None
    if e is not None:
        pe=np.linalg.norm(e[:,:3],axis=1); re=np.linalg.norm(e[:,3:],axis=1)
        if np.max(pe)>config['position_tolerance_m']: reasons.append('target_position')
        if np.max(re)>config['orientation_tolerance_rad']: reasons.append('target_orientation')
    collision=checker.check(robot.fk(q,base),config['self_clearance_m'])
    if collision['collision_pairs']: reasons.append('mesh_surface_intersection')
    if collision['below_clearance']: reasons.append('mesh_clearance')
    return dict(status='REJECTED' if reasons else 'ACCEPTED_CONDITIONAL',reasons=reasons,
        joint_limit_violations=violations,collision=collision,
        position_errors_m=None if e is None else pe.tolist(),
        orientation_errors_rad=None if e is None else re.tolist())


def summary(records):
    from collections import Counter
    return dict(count=len(records), statuses=dict(Counter(r['status'] for r in records)),
                reasons=dict(Counter(reason for r in records for reason in r['reasons'])))


def run(config_path,output,seed=None):
    started=time.perf_counter()
    out=Path(output)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose an empty output directory; results are not overwritten')
    out.mkdir(parents=True,exist_ok=True)
    raw_config=Path(config_path).read_text(); config=yaml.safe_load(raw_config)
    if seed is not None: config['seed']=seed
    rng=np.random.default_rng(config['seed'])
    path=unpack(config['archive'],config['asset_directory'])
    asset=Asset(path); audit=asset.audit(); dump(out/'asset_audit.json',audit)
    robot=Robot(asset); checker=CollisionChecker(asset)
    (out/'kinematics_only.urdf').write_text(robot.adapted_xml)
    dump(out/'collision_policy.json',dict(checked=checker.pairs,excluded=checker.excluded,
         missing_environment=['floor','bed','human','gripper','handle'],
         mesh_occupancy_caveat='open meshes; surface clearance only, no closed-solid certification'))
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    reference=np.zeros(len(robot.names))
    for n,value in config['reference_joint_positions_rad'].items(): reference[robot.index[n]]=value
    targets=[robot.fk(reference)[n] for n in TIPS]
    reference_check=evaluate(robot,checker,reference,np.zeros(3),config,targets)
    dump(out/'targets.json',dict(frame='world: aligned with reference base_link; z=0 NOT ground',
        tip_names=TIPS,handle_transform='identity on flange; explicit virtual fixture',
        reference_q_rad=reference.tolist(),world_targets=[t.tolist() for t in targets],check=reference_check))
    # Full-limit random sampling; wheels and joints 5/6 held at 0 for this experiment.
    random_records=[]; feasible=[]; derivative=[]
    for i in range(config['random_attempt_budget']):
        q=np.zeros(len(robot.names)); q[robot.active]=rng.uniform(robot.lower[robot.active],robot.upper[robot.active])
        base=np.array([rng.uniform(*config[k]) for k in ['base_x_m','base_y_m','base_yaw_rad']])
        record=dict(id=f'v{i:04d}',q_rad=q.tolist(),base_xy_yaw=base.tolist())
        try:
            record.update(evaluate(robot,checker,q,base,config))
            if record['status']=='ACCEPTED_CONDITIONAL':
                for eps in config['finite_difference_steps_rad']:
                    check=robot.differential_check(q,base,eps)
                    derivative.append(dict(id=record['id'],**check))
                record['differential_check']=derivative[-2]
                feasible.append(record)
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            record.update(status='NUMERICAL_ERROR',reasons=[type(exc).__name__],error=str(exc))
        random_records.append(record)
        if len(feasible)>=config['random_validation_count']: break
    dump(out/'random_validation.json',random_records)
    dump(out/'differential_checks.json',derivative)
    # Multistart bounded local least squares; local failure is not global infeasibility.
    candidates=[]
    for i in range(config['search_count']):
        t0=time.perf_counter()
        base=np.zeros(3) if i<config['search_count']//4 else np.array([
            rng.uniform(*config[k]) for k in ['base_x_m','base_y_m','base_yaw_rad']])
        q0=reference.copy()
        q0[robot.active]+=rng.normal(0,config['ik_seed_std_rad'],len(robot.active))
        q0=np.clip(q0,robot.lower+1e-9,robot.upper-1e-9)
        rec=dict(id=f'ik{i:04d}',initial_q_rad=q0.tolist(),base_xy_yaw=base.tolist())
        try:
            q,solver=robot.ik(targets,base,q0,config['ik_max_nfev'],config['orientation_scale_m'])
            rec.update(q_rad=q.tolist(),solver=solver,**evaluate(robot,checker,q,base,config,targets))
        except (ValueError,RuntimeError,FloatingPointError) as exc:
            rec.update(status='NUMERICAL_ERROR',reasons=[type(exc).__name__],error=str(exc))
        rec['elapsed_s']=time.perf_counter()-t0
        candidates.append(rec)
        # Preserve partial outcomes even if an external process interruption occurs.
        dump(out/'candidates.json',candidates)
        print(rec['id'],rec['status'],rec['reasons'],flush=True)
    accepted=[r for r in candidates if r['status']=='ACCEPTED_CONDITIONAL']
    with (out/'candidates.csv').open('w') as f:
        fields=['id','status','reasons','base_x_m','base_y_m','base_yaw_rad','position_max_m','rotation_max_rad','elapsed_s']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in candidates:
            writer.writerow(dict(id=r['id'],status=r['status'],reasons=';'.join(r['reasons']),
                **dict(zip(fields[3:6],r['base_xy_yaw'])),
                position_max_m=max(r['position_errors_m']) if r.get('position_errors_m') else None,
                rotation_max_rad=max(r['orientation_errors_rad']) if r.get('orientation_errors_rad') else None,
                elapsed_s=r['elapsed_s']))
    def git(*args): return subprocess.check_output(['git',*args],text=True).strip()
    metric_keys=['fk_position_max_m','fk_rotation_max_rad','jacobian_linear_max_abs','jacobian_angular_max_abs']
    stats={k:max(d[k] for d in derivative) for k in metric_keys} if derivative else {}
    metrics=dict(random=summary(random_records),ik=summary(candidates),validation_maxima=stats,
        target_position_max_m=max((max(r['position_errors_m']) for r in accepted),default=None),
        target_orientation_max_rad=max((max(r['orientation_errors_rad']) for r in accepted),default=None),
        distinct_base_count=len(set(tuple(r['base_xy_yaw']) for r in accepted)),
        reference_status=reference_check['status'],elapsed_s=time.perf_counter()-started)
    dump(out/'summary.json',metrics)
    provenance=dict(source_commit=git('rev-parse','HEAD'),git_status=git('status','--porcelain','--untracked-files=no'),
        seed=config['seed'],platform=platform.platform(),python=platform.python_version(),
        versions={n:metadata.version(n) for n in ['mujoco','python-fcl','trimesh','numpy','scipy','matplotlib','pyyaml','pytest']},
        archive_sha256=sha256(config['archive']),config_source_sha256=sha256(config_path),
        units=dict(position='m',angle='rad',time='s'),scope='M1-A/B static geometry; no dynamics')
    dump(out/'metadata.json',provenance)
    np.savez_compressed(out/'configurations.npz',joint_names=np.array(robot.names),
        all_q_rad=np.array([r.get('q_rad',[np.nan]*len(robot.names)) for r in candidates]),
        initial_q_rad=np.array([r['initial_q_rad'] for r in candidates]),
        base_xy_yaw=np.array([r['base_xy_yaw'] for r in candidates]),
        status=np.array([r['status'] for r in candidates]),world_targets=np.array(targets))
    print(json.dumps(metrics,indent=2),flush=True)
    return metrics
