"""M1 geometry checks; assets required for M1 tests, never download on import."""
from pathlib import Path
import sys
import zipfile
import numpy as np
import pytest
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
pytest.importorskip('mujoco')
pytest.importorskip('fcl')
from h1_kinematics.assets import Asset,unpack,ARCHIVE_SHA256,sha256
from h1_kinematics.model import Robot,TIPS
from h1_kinematics.collision import CollisionChecker
from h1_kinematics.experiment import evaluate

@pytest.fixture(scope='module')
def setup_model():
    archive=Path('assets/h1/source.zip')
    if not archive.exists(): pytest.fail('Supply audited archive to assets/h1/source.zip for M1 tests')
    a=Asset(unpack(archive,'assets/h1/original'))
    r=Robot(a); c=CollisionChecker(a); config=yaml.safe_load(Path('configs/m1_kinematics.yaml').read_text())
    q=np.zeros(len(r.names))
    for n,v in config['reference_joint_positions_rad'].items(): q[r.index[n]]=v
    return a,r,c,config,q

def test_archive_integrity_and_complete_mesh_resolution(setup_model):
    a,r,c,config,q=setup_model
    assert sha256('assets/h1/source.zip')==ARCHIVE_SHA256
    assert len(a.links)==34 and len(a.joints)==33
    assert r.model.nq==24
    assert len(c.meshes)==32
    for l in a.links.values():
        for e in l.findall('.//mesh'): assert a.mesh_path(e.get('filename')).exists()
    with pytest.raises(ValueError): a.mesh_path('package://missing/absent.stl')

def test_source_inertias_not_replaced(setup_model):
    a,r,*_=setup_model
    import xml.etree.ElementTree as E
    adapted=E.fromstring(r.adapted_xml)
    for l in adapted.findall('link'):
        source=a.links[l.get('name')].find('inertial'); target=l.find('inertial')
        assert (source is None)==(target is None)
        if source is not None: assert E.tostring(source)==E.tostring(target)

def test_library_fk_and_bilateral_jacobian_vs_independent_difference(setup_model):
    a,r,c,config,q=setup_model
    rng=np.random.default_rng(33)
    for _ in range(5):
        sample=q.copy(); sample[r.active]=rng.uniform(r.lower[r.active],r.upper[r.active])
        for eps in [1e-5,1e-6,1e-7]:
            check=r.differential_check(sample,[.02,-.03,.08],eps)
            assert check['fk_position_max_m']<1e-12
            assert check['fk_rotation_max_rad']<1e-12
            assert check['jacobian_linear_max_abs']<1e-7
            assert check['jacobian_angular_max_abs']<1e-7
    jl,jr=r.jacobians(q)
    assert np.max(abs(jl[:,[r.dof[n] for n in r.names if '08R' in n]]))==0
    assert np.max(abs(jr[:,[r.dof[n] for n in r.names if '08L' in n]]))==0
    assert np.max(abs(jl[:,:4]))==0 and np.max(abs(jr[:,:4]))==0

def test_collision_reference_narrow_phase_and_clearance_detection(setup_model):
    a,r,c,config,q=setup_model
    result=c.check(r.fk(q))
    assert result['passed']
    assert any('link5' in pair[0] and 'link7' in pair[1] for pair in result['narrow_phase_pairs'])
    # Deliberately overlap both arm link3 meshes: tests included left/right arm pair.
    frames=r.fk(q); left='AR5-5_08L-W4C1C9-ZY2_link3'; right=left.replace('08L','08R')
    frames[right]=frames[left].copy()
    result=c.check(frames)
    assert not result['passed'] and [left,right] in result['collision_pairs']
    assert not c.check(r.fk(q),clearance_m=.02)['passed']

def test_dual_arm_ik_preserves_both_world_poses_with_changed_base(setup_model):
    a,r,c,config,q=setup_model
    targets=[r.fk(q)[tip] for tip in TIPS]
    base=np.array([.03,-.02,.04]); seed=q.copy()
    seed[r.active]+=np.random.default_rng(20).normal(0,.15,len(r.active))
    result,solver=r.ik(targets,base,seed)
    check=evaluate(r,c,result,base,config,targets)
    assert max(check['position_errors_m'])<1e-5
    assert max(check['orientation_errors_rad'])<1e-5
    assert np.all(result>=r.lower-1e-10) and np.all(result<=r.upper+1e-10)
    assert np.linalg.norm(result-q)>.02

def test_bad_inputs_and_local_target_failure_not_success(setup_model):
    a,r,c,config,q=setup_model
    with pytest.raises(ValueError): r.fk(q*np.nan)
    with pytest.raises(ValueError): r.fk(q,base=[0,0])
    targets=[r.fk(q)[tip] for tip in TIPS]; targets[0][:3,3]+=[10,0,0]
    result,solver=r.ik(targets,np.zeros(3),q,max_nfev=3)
    check=evaluate(r,c,result,np.zeros(3),config,targets)
    assert check['status']=='REJECTED' and 'target_position' in check['reasons']
    bad=q.copy();bad[r.index['1-Joint']]=1.0
    assert 'joint_limit' in evaluate(r,c,bad,np.zeros(3),config)['reasons']


def test_repeated_asset_import_is_read_only_and_detects_corruption(tmp_path):
    path=unpack('assets/h1/source.zip',tmp_path/'assets')
    stamp=path.stat().st_mtime_ns
    unpack('assets/h1/source.zip',tmp_path/'assets')
    assert path.stat().st_mtime_ns==stamp
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='differs from archive'):
        unpack('assets/h1/source.zip',tmp_path/'assets')
