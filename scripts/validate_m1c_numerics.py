"""Reproduce numerical error statistics reported for the M1-C overlay."""
from pathlib import Path
import argparse
import copy
import json
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from h1_capability.experiment import setup
from h1_capability.model import CapabilityModel
from h1_kinematics.assets import dump


def run(output):
    out=Path(output);c=yaml.safe_load((out/'config.yaml').read_text());asset,_,_=setup(c)
    records=json.loads((out/'force_and_instantaneous.json').read_text())
    r=next(r for r in records if r['status']=='ACCEPTED');q=np.array(r['q_rad']);b=np.array(r['base_xy_yaw'])
    overlay=copy.deepcopy(c['model'])
    overlay['handle_frame']['left']={'xyz_m':[.03,.01,.08],'rpy_rad':[.1,.2,.3]}
    m=CapabilityModel(asset,overlay);jac=m.jacobians(q,b,True)[0];worst=0.;epsilon=1e-6
    for col in m.robot.active+list(range(24,27)):
        qa=q.copy();qb=q.copy();ba=b.copy();bb=b.copy()
        if col<24:qa[col]+=epsilon;qb[col]-=epsilon
        else:ba[col-24]+=epsilon;bb[col-24]-=epsilon
        ta=m.poses(qa,ba)[0];tb=m.poses(qb,bb)[0]
        difference=np.r_[(ta[:3,3]-tb[:3,3])/(2*epsilon),
            Rotation.from_matrix(ta[:3,:3]@tb[:3,:3].T).as_rotvec()/(2*epsilon)]
        worst=max(worst,float(np.max(abs(difference-jac[:,col]))))
    force=[x for r in records if r['status']=='ACCEPTED' for ks in r['force'].values() for ds in ks.values() for x in ds.values()]
    answer=dict(offset_handle_and_base_jacobian_max_abs_error=worst,epsilon=epsilon,
        lp_max_torque_violation_nm=max(x['max_violation_nm'] for x in force),
        lp_max_equalities_n=max(x['max_equality_error_n'] for x in force))
    assert worst<2e-9 and answer['lp_max_torque_violation_nm']<1e-6 and answer['lp_max_equalities_n']<1e-6
    dump(out/'numerical_checks.json',answer);print(json.dumps(answer,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');run(parser.parse_args().output)
