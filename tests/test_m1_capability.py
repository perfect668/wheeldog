from pathlib import Path
import sys
import copy
import numpy as np
import pytest
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
pytest.importorskip('mujoco');pytest.importorskip('fcl')
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicHermiteSpline
from h1_kinematics.assets import Asset,unpack
from h1_kinematics.collision import CollisionChecker
from h1_capability.model import CapabilityModel,DIRECTIONS
from h1_capability.force import ray_capacity
from h1_capability.instantaneous import cooperative_velocity
from h1_capability.reachability import construct_path,spline_extrema,validate_geometry
from h1_capability.pairing import match_pairs

@pytest.fixture(scope='module')
def setup_c():
    import json
    c=yaml.safe_load(Path('configs/m1_capability.yaml').read_text())
    a=Asset(unpack(c['archive'],c['asset_directory']));m=CapabilityModel(a,c['model']);k=CollisionChecker(a)
    r=next(x for x in json.loads(Path(c['candidates']).read_text()) if x['status']=='ACCEPTED_CONDITIONAL')
    return c,a,m,k,np.array(r['q_rad']),np.array(r['base_xy_yaw'])

def test_parameter_overlay_chain_mapping_units_and_source_immutable(setup_c):
    c,a,m,k,q,b=setup_c
    for names,idx in zip(m.arm_names,m.arm_indices):
        assert [n.rsplit('_',1)[1] for n in names]==list(map(str,range(1,8)))
        assert m.robot.upper[idx[-1]]==pytest.approx(np.pi/2)
        assert float(a.joints[names[-1]].find('limit').get('upper'))==1.0472
    assert m.velocity[0]==pytest.approx(np.deg2rad(175))
    assert np.isinf(m.budgets()[m.torso_indices]).all()

def test_contact_sign_virtual_work_and_shared_torso_sum(setup_c):
    c,a,m,k,q,b=setup_c
    force=np.array([10,5,20,-2,3,15.]);v=np.random.default_rng(1).normal(size=24)
    A=m.contact_map(q,b);jl,jr=m.jacobians(q,b)
    assert np.dot(A@force,v)==pytest.approx(np.dot(force,np.r_[jl[:3]@v,jr[:3]@v]))
    both=A@force
    assert np.allclose(both[m.torso_indices],(jl[:3].T@force[:3]+jr[:3].T@force[3:])[m.torso_indices])
    reaction=-A@force
    assert np.allclose(reaction+both,0)

def test_handle_offset_and_base_jacobian_difference(setup_c):
    c,a,m,k,q,b=setup_c
    config=copy.deepcopy(c['model']);config['handle_frame']['left']={'xyz_m':[.03,.01,.08],'rpy_rad':[.1,.2,.3]}
    model=CapabilityModel(a,config);j=model.jacobians(q,b,True)[0]
    for col in model.robot.active+list(range(24,27)):
        eps=1e-6;qa=q.copy();qb=q.copy();ba=b.copy();bb=b.copy()
        if col<24:qa[col]+=eps;qb[col]-=eps
        else:ba[col-24]+=eps;bb[col-24]-=eps
        ta=model.poses(qa,ba)[0];tb=model.poses(qb,bb)[0]
        fd=np.r_[(ta[:3,3]-tb[:3,3])/(2*eps),Rotation.from_matrix(ta[:3,:3]@tb[:3,:3].T).as_rotvec()/(2*eps)]
        assert np.allclose(fd,j[:,col],atol=2e-9)

def test_force_lp_exact_solution_infeasible_unbounded_and_shared_limits():
    A=np.vstack([np.c_[np.eye(3),np.zeros((3,3))],np.c_[np.zeros((3,3)),np.eye(3)]])
    answer=ray_capacity(A,np.ones(6)*10,np.array([1.,0,0]))
    assert answer['capacity_n']==pytest.approx(20)
    assert ray_capacity(A,np.ones(6)*10,np.array([1.,0,0]),task=[11,0,0,0,0,0])['status']=='TASK_INFEASIBLE'
    assert ray_capacity(np.zeros((1,6)),[1],[1,0,0])['status']=='UNBOUNDED'
    # Shared joint sees both loads; cannot grant full independent budget twice.
    shared=np.array([[1.,0,0,1,0,0]])
    assert ray_capacity(shared,[10],[1,0,0])['capacity_n']==pytest.approx(10)
    assert ray_capacity(A,np.ones(6)*20,[1,0,0])['capacity_n']>=answer['capacity_n']

def test_instantaneous_cooperative_twist_and_mobile_superset(setup_c):
    c,a,m,k,q,b=setup_c;p=c['profiles']['nominal']
    fixed=cooperative_velocity(m,q,b,DIRECTIONS['left'],p,False)
    mobile=cooperative_velocity(m,q,b,DIRECTIONS['left'],p,True)
    assert fixed['status']==mobile['status']=='OPTIMAL'
    assert mobile['speed_m_s']>=fixed['speed_m_s']-1e-8
    j=m.task_jacobian(q,b,True);v=np.array(mobile['coordinate_velocity'])
    assert np.allclose(j@v,np.tile([0,mobile['speed_m_s'],0,0,0,0],2),atol=1e-7)
    assert np.linalg.norm(v[-3:-1])<=1.5+1e-8

def test_spline_extrema_catch_between_node_speed_acceleration():
    t=np.array([0.,1.]);x=np.array([[0.,0.,0.],[1.,0.,0.]])
    spline=CubicHermiteSpline(t,x,np.zeros_like(x))
    ext=spline_extrema(spline,t)
    assert ext['v_abs_max'][0]==pytest.approx(1.5)
    assert ext['a_abs_max'][0]==pytest.approx(6.)
    assert ext['q_min'][0]==0 and ext['q_max'][0]==1

def test_finite_path_delay_accel_pose_and_force_constraints(setup_c):
    c,a,m,k,q,b=setup_c;profile=c['profiles']['nominal'];motion=c['motion']
    task=np.r_[c['force']['task_left_n'],c['force']['task_right_n']]
    r=construct_path(m,q,b,DIRECTIONS['left'],.005,.3,profile,False,motion,'speed_weighted')
    assert r['status']=='NEEDS_COLLISION_CHECK'
    assert np.max(abs(r['spline'](.02)-q[m.active]))<1e-12
    assert np.max(abs(r['spline'](.04,1)))<1e-12
    assert np.max(abs(r['spline'](.3,1)))<1e-12
    assert validate_geometry(m,k,r,q,b,DIRECTIONS['left'],.3,profile,False,motion,task)['status']=='PATH_ACCEPTED'
    weak=copy.deepcopy(profile);weak['joint_acceleration_rad_s2']=.0001
    r=construct_path(m,q,b,DIRECTIONS['left'],.005,.3,weak,False,motion,'speed_weighted')
    assert 'joint_or_yaw_acceleration' in r['reasons']
    r=construct_path(m,q,b,DIRECTIONS['left'],.005,.02,profile,False,motion,'speed_weighted')
    assert r['reasons']==['delay_exhausts_window']

def test_pairing_uses_whole_force_vector_and_is_symmetric():
    a=dict(id='a',status='ACCEPTED',force_match_vector_n=[100.]*12)
    b=dict(id='b',status='ACCEPTED',force_match_vector_n=[100.]*11+[125.])
    assert match_pairs([a,b])[0]['status']=='REJECTED'
    b['force_match_vector_n'][-1]=105
    assert match_pairs([a,b])[0]['status']=='MATCHED'
    assert match_pairs([b,a])[0]['max_relative_gap']==match_pairs([a,b])[0]['max_relative_gap']

def test_search_returns_saved_positive_path_and_mobile_retains_fixed_option(setup_c,tmp_path):
    from h1_capability.reachability import search
    from h1_capability.experiment import include_fixed_option
    c,a,m,k,q,b=setup_c;options=copy.deepcopy(c['motion'])
    options.update(max_displacement_m=.005,bisection_iterations=2)
    task=np.r_[c['force']['task_left_n'],c['force']['task_right_n']]
    path=tmp_path/'path.npz'
    result=search(m,k,q,b,DIRECTIONS['left'],.3,c['profiles']['nominal'],False,options,task,path)
    assert result['status']=='VERIFIED_PATH_FOUND' and result['achieved_displacement_m']>0
    with np.load(path) as data:
        assert float(data['displacement_m'])==result['achieved_displacement_m']
        assert np.allclose(data['slopes'][0],0) and np.allclose(data['slopes'][-1],0)
    fixed=dict(experiment='main',candidate='a',direction='left',horizon_s=.3,mobile=False,
               status='VERIFIED_PATH_FOUND',achieved_displacement_m=.02,path_file='verified-fixed.npz')
    mobile=dict(fixed,mobile=True,achieved_displacement_m=.01,path_file='local-mobile.npz')
    union=include_fixed_option([fixed,mobile])[-1]
    assert union['achieved_displacement_m']==.02 and union['path_file']=='verified-fixed.npz'
    assert union['local_mobile_search']['achieved_displacement_m']==.01


def test_saved_path_replay_detects_corrupted_boundary_slope(setup_c,tmp_path):
    from h1_capability.reachability import search
    from h1_capability.verification import initialize,replay
    c,a,m,k,q,b=setup_c;options=copy.deepcopy(c['motion'])
    options.update(max_displacement_m=.005,bisection_iterations=1)
    path=tmp_path/'replay.npz';task=np.r_[c['force']['task_left_n'],c['force']['task_right_n']]
    result=search(m,k,q,b,DIRECTIONS['left'],.3,c['profiles']['nominal'],False,options,task,path)
    row=dict(result,case_id='test',candidate='test',experiment='main',path_file=str(path),horizon_s=.3)
    initialize(c)
    assert replay(row,True)['status']=='REPLAY_PASSED'
    with np.load(path,allow_pickle=False) as data:corrupt={name:data[name] for name in data.files}
    corrupt['slopes'][0,0]=100.
    np.savez_compressed(path,**corrupt)
    failed=replay(row,False)
    assert failed['status']=='REPLAY_FAILED'
    assert 'rest_boundary' in failed['reasons'] and 'velocity' in failed['reasons']
