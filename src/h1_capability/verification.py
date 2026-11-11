"""Replay saved spline artifacts, including denser task and mesh sampling.

This is numerical consistency checking of an offline path. It does not turn
sampled collision checks into a continuous collision certificate.
"""
from pathlib import Path
import argparse
import copy
import json
import concurrent.futures
import subprocess
import time
import numpy as np
import yaml
from scipy.interpolate import CubicHermiteSpline
from h1_kinematics.assets import dump,sha256
from .model import CapabilityModel
from .experiment import setup,frozen_inventory
from .reachability import spline_extrema,validate_geometry
from .analysis import write_csv,describe

STATE=None


def initialize(config):
    global STATE
    STATE=(config,*setup(config))


def replay(row,geometry=False,substeps=8):
    c,asset,model0,checker=STATE;began=time.perf_counter()
    model_config=copy.deepcopy(c['model'])
    if 'joint7_deg' in row:model_config['joint_ranges_deg'][6]=[-row['joint7_deg'],row['joint7_deg']]
    model=model0 if model_config==c['model'] else CapabilityModel(asset,model_config)
    profile=copy.deepcopy(c['profiles'][row.get('profile','nominal')]);profile.update(row.get('profile_overrides',{}))
    options=copy.deepcopy(c['motion']);options.update(row.get('motion_overrides',{}));options['collision_substeps']=substeps
    result=dict(case_id=row['case_id'],experiment=row['experiment'],candidate=row['candidate'],
        path_file=row['path_file'],geometry_checked=geometry,collision_substeps=substeps if geometry else None,
        reasons=[],status='REPLAY_PASSED')
    with np.load(row['path_file'],allow_pickle=False) as data:
        times=data['times_s'];x=data['coordinates'];slopes=data['slopes'];q0=data['q0_rad'];base0=data['base0_xy_yaw']
        mobile=bool(data['mobile']);direction=data['direction_world'];h=float(data['horizon_s']);delay=float(data['delay_s'])
        distance=float(data['displacement_m']);initial=data['initial_handle_poses']
        if not all(np.isfinite(z).all() for z in [times,x,slopes,q0,base0,direction,initial]):result['reasons'].append('nonfinite_data')
        if not np.array_equal(data['active_indices'],model.active):result['reasons'].append('active_coordinate_mapping')
    spline=CubicHermiteSpline(times,x,slopes);ext=spline_extrema(spline,times);n=len(model.active)
    if abs(distance-row['achieved_displacement_m'])>1e-12:result['reasons'].append('json_npz_displacement')
    if abs(h-row['horizon_s'])>1e-12 or abs(delay-profile['delay_s'])>1e-12:result['reasons'].append('time_metadata')
    result['peak_velocity_ratio']=float(max(ext['v_abs_max']/model.coordinate_velocity(profile,mobile)))
    result['peak_acceleration_ratio']=float(max(ext['a_abs_max']/model.coordinate_acceleration(profile,mobile)))
    if result['peak_velocity_ratio']>1+1e-8:result['reasons'].append('velocity')
    if result['peak_acceleration_ratio']>1+1e-8:result['reasons'].append('acceleration')
    if (ext['q_min'][:n]<model.robot.lower[model.active]-1e-9).any() or (ext['q_max'][:n]>model.robot.upper[model.active]+1e-9).any():result['reasons'].append('position')
    if mobile and ext['base_speed_norm_max']>model.config['base_velocity_m_s']+1e-8:result['reasons'].append('base_speed')
    if mobile and ext['base_accel_norm_max']>profile['base_acceleration_m_s2']+1e-8:result['reasons'].append('base_acceleration')
    if np.max(abs(spline(0,1)))>1e-10 or np.max(abs(spline(h,1)))>1e-10:result['reasons'].append('rest_boundary')
    before=np.linspace(0,delay,7)
    if np.max(abs(spline(before)-x[0]))>1e-10:result['reasons'].append('delay_not_stationary')
    # Independent dense derivative evaluation cross-checks polynomial extrema.
    samples=np.unique(np.concatenate([np.linspace(a,b,13) for a,b in zip(times[:-1],times[1:])]))
    for degree,key in [(1,'v_abs_max'),(2,'a_abs_max')]:
        if np.max(np.max(abs(spline(samples,degree)),axis=0)-ext[key])>1e-7:result['reasons'].append('analytic_extrema_disagreement')
    if geometry:
        path=dict(times=times,spline=spline,displacement_m=distance,initial_poses=initial)
        task=np.r_[c['force']['task_left_n'],c['force']['task_right_n']]
        check=validate_geometry(model,checker,path,q0,base0,direction,h,profile,mobile,options,task)
        result['geometry']=check
        if check['status']!='PATH_ACCEPTED':result['reasons'].append(check['status'])
    if result['reasons']:result['status']='REPLAY_FAILED'
    result['path_sha256']=sha256(row['path_file']);result['elapsed_s']=time.perf_counter()-began
    return result


def replay_job(job):return replay(*job)


def run(output,workers=4):
    out=Path(output);c=yaml.safe_load((out/'config.yaml').read_text())
    rows=json.loads((out/'reachability.json').read_text())
    if (out/'sensitivity.json').exists():rows+=json.loads((out/'sensitivity.json').read_text())
    pairs=json.loads((out/'selected_pairs.json').read_text());ids={x for p in pairs for x in [p['a'],p['b']]}
    jobs=[(row,row['candidate'] in ids,8) for row in rows if row['status']=='VERIFIED_PATH_FOUND']
    start=time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(c,)) as pool:
        results=list(pool.map(replay_job,jobs))
    dump(out/'replay.json',results)
    write_csv(out/'replay.csv',[{k:v for k,v in r.items() if k not in ['geometry','reasons']}|{'reasons':';'.join(r['reasons'])} for r in results])
    passed=[r for r in results if r.get('geometry',{}).get('status')=='PATH_ACCEPTED']
    summary=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        rows=len(rows),paths_replayed=len(results),geometry_checked=sum(r['geometry_checked'] for r in results),
        failures=[r for r in results if r['status']!='REPLAY_PASSED'],elapsed_s=time.perf_counter()-start,
        max_peak_velocity_ratio=max(r['peak_velocity_ratio'] for r in results),
        max_peak_acceleration_ratio=max(r['peak_acceleration_ratio'] for r in results),
        dense_position_error_m=describe([r['geometry']['max_position_error_m'] for r in passed]),
        dense_rotation_error_rad=describe([r['geometry']['max_rotation_error_rad'] for r in passed]),
        dense_minimum_clearance_m=min((r['geometry']['min_sampled_mesh_distance_m'] for r in passed),default=None),
        frozen_integrity=frozen_inventory(),
        interpretation='Saved-path consistency plus denser numerical sampling, not causal feedback or continuous solid collision certification')
    dump(out/'replay_summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.output,a.workers)
