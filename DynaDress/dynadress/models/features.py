"""Versioned observable state/action interface; attributes are a separate input.

Features never use future state, human holding effort or outcomes. Missing load
is rejected, not imputed to zero. Full raw cuff state remains in each episode.
"""
import numpy as np

SCHEMA='m2.features.v1'
HISTORY=8
DT=.05
RING_SELECTION=np.arange(0,80,10)


def pose_feature(matrix):
    m=np.asarray(matrix)
    return np.r_[m[:3,3],m[:3,:2].reshape(-1)]


def schema(robot_names):
    groups={};names=[];floors=[]
    def add(group, labels, unit, floor):
        start=len(names);names.extend([f'{group}.{n} [{unit}]' for n in labels])
        floors.extend([floor]*len(labels));groups[group]=[start,len(names)]
    add('human_q',range(4),'rad',.015)
    add('human_dq',range(4),'rad/s',.03)
    add('robot_q',robot_names,'rad; jaws m',.01)
    add('robot_dq',robot_names,'rad/s; jaws m/s',.03)
    add('ee_pose',[f'{s}_{v}' for s in 'LR' for v in ['x','y','z','r00','r01','r10','r11','r20','r21']],'m / rotation',.01)
    add('cuff_center',['x','y','z'],'m',.005)
    add('cuff_normal',['x','y','z'],'unitless',.02)
    add('cuff_spans',['major','minor'],'m',.005)
    add('cuff_area',['area'],'m2',.0002)
    add('cuff_planarity',['rms'],'m',.001)
    add('progress',['axial_from_wrist'],'m',.01)
    add('cuff_points',[f'{i}_{c}' for i in RING_SELECTION for c in 'xyz'],'m',.005)
    add('load',[f'{s}_{v}' for s in 'LR' for v in ['Fx','Fy','Fz','Tx','Ty','Tz']],'N / Nm',.1)
    for s in range(groups['load'][0],groups['load'][1]):
        if (s-groups['load'][0])%6>=3:floors[s]=.01
    action_names=[f'{s}.target_{v}' for s in 'LR' for v in ['x','y','z','r00','r01','r10','r11','r20','r21']]
    action_names += [f'{s}.delta_{v}' for s in 'LR' for v in 'xyz']+[f'{s}.rot_delta_{v}' for s in 'LR' for v in 'xyz']
    return dict(version=SCHEMA,dt_s=DT,history_samples=HISTORY,state_names=names,groups=groups,state_std_floor=floors,
                action_names=action_names,action_std_floor=[.01]*18+[.0002]*6+[.0005]*6,
                condition_names=[f'K{i}_Nm_rad' for i in range(4)]+[f'D{i}_Nm_s_rad' for i in range(4)]+['friction']+[f'q_ref{i}_rad' for i in range(4)],
                condition_std_floor=[1.]*8+[.05]+[.01]*4,
                load_source='Newton-Euler external wrist residual, not isolated contact or physical F/T sensor')


def extract(row):
    st=row['solver_state']['value'];g=st['geometry'];load=row['estimated_load']
    if g is None or st['cuff_keypoints_m'] is None or not load['available']:
        return None
    a=row['action'];ring=np.asarray(st['cuff_keypoints_m'])
    if ring.shape != (80,3):raise ValueError('Unexpected pinned cuff topology')
    state=np.concatenate([np.asarray(st['human_q_rad']),np.asarray(st['human_dq_rad_s']),np.asarray(st['robot_q']),np.asarray(st['robot_dq']),
        *[pose_feature(st['end_effector_actual'][s]) for s in 'LR'],
        g['center_m'],g['normal'],g['spans_m'],[g['area_m2'],g['planarity_rms_m'],g['axial_from_wrist_m']],ring[RING_SELECTION].ravel(),
        *[load['value']['external_world_N_Nm'][s] for s in 'LR']]).astype(np.float32)
    action=np.concatenate([*[pose_feature(a['end_effector_targets'][s]) for s in 'LR'],
                 *[a['end_effector_delta_world_m'][s] for s in 'LR'],
                 *[a['rotation_delta_world_rad'][s] for s in 'LR']]).astype(np.float32)
    if not np.isfinite(np.r_[state,action]).all():raise ValueError('Nonfinite model feature')
    return state,action


def attributes(config):
    """First nine entries are properties for a future estimator; q_ref stays configured."""
    h=config['human']
    return np.r_[h['stiffness'],h['damping'],config['garment']['friction'],h['reference']].astype(np.float32)
