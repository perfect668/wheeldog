import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from dynadress.human.passive_arm import Impedance
from dynadress.recording.episode import Episode
from dynadress.robots.h1 import H1Kinematics

class Contracts(unittest.TestCase):
    def test_impedance_gravity_equilibrium_and_limit(self):
        imp=Impedance();g=np.array([1.,2.,3.,4.]);tau,_=imp.effort(imp.reference,np.zeros(4),g)
        np.testing.assert_allclose(tau,g)
        tau,_=imp.effort(np.array(imp.reference)+100,np.zeros(4),g)
        self.assertTrue(np.all(np.abs(tau)<=imp.torque_limit))
        with self.assertRaises(FloatingPointError):imp.effort([np.nan]*4,[0]*4,[0]*4)
    def test_missing_measurement_and_bad_clock(self):
        with tempfile.TemporaryDirectory() as d:
            episode=Episode(d,{'scope':'unit test'})
            episode.write(0,0,0,solver_state={'human_q_rad':[0]*4})
            with self.assertRaises(ValueError):episode.write(0,0,0)
            episode.close('unit_test','done')
            row=json.loads((Path(d)/'samples.jsonl').read_text())
            self.assertEqual(row['solver_contact'],{'available':False,'value':None})
            self.assertEqual(row['measured_wrench'],{'available':False,'value':None})
    def test_supplied_asset_reachable_ik_and_velocity_bound(self):
        path=Path('assets/h1/generated/h1_task.urdf')
        if not path.exists():self.skipTest('requires locally prepared supplied asset')
        model=H1Kinematics(path);q={n:(.5 if n.endswith('_4') else 0.) for n in model.names}
        for side in 'LR':
            target=model.fk(q,'tool_'+side);target[0,3]+=.025;target[1,3]+=(-.025 if side=='L' else .025);target[2,3]+=.01
            goal,err=model.solve(side,target,q)
            self.assertLess(err['position_error_m'],1e-4);self.assertLess(err['rotation_error_rad'],1e-3)
            cmd=model.bounded_step(q,goal,1/240)
            self.assertLessEqual(max(abs(cmd[n]-q[n]) for n in q),.25/240+1e-12)

if __name__=='__main__':unittest.main()
