"""Numerical checks for load compensation and data provenance boundaries."""
import copy,json,unittest
from pathlib import Path
import numpy as np
from dynadress.interaction.wrist_load import inertial_support
from dynadress.interaction.excitation import command
from dynadress.models.features import extract,schema


class M2Contracts(unittest.TestCase):
    def test_subtree_weight_and_moment(self):
        # Three bodies: omitting the two 40 g fingers creates a 0.7848 N bias.
        m=np.array([.53,.04,.04]);com=np.array([[0,0,0],[.1,0,0],[.1,0,0]])
        support=inertial_support(m,com,np.zeros((3,3,3)),np.zeros((3,6)),np.zeros((3,6)),np.zeros(3))
        np.testing.assert_allclose(support,[0,0,.61*9.81,0,-.08*.1*9.81,0],atol=1e-10)

    def test_acceleration_compensation(self):
        # A body translating at 2 m/s^2 requires an additional 4 N, not load.
        acc=np.array([[2.,0,0,0,0,0]])
        support=inertial_support(np.array([2.]),np.array([[0,1.,0]]),np.zeros((1,3,3)),np.zeros((1,6)),acc,np.zeros(3),gravity=[0,0,0])
        np.testing.assert_allclose(support,[4,0,0,0,0,-4])

    def test_excitation_is_bounded_and_returns(self):
        for name in ['baseline','probe_a','probe_b','probe_c']:
            rows=[command(name,t) for t in np.linspace(0,20,1000)]
            self.assertLessEqual(max(abs(r['roll_rad']) for r in rows),np.radians(2))
            self.assertLessEqual(max(abs(r['differential_m']) for r in rows),.002)
            self.assertEqual(rows[-1]['roll_rad'],0)
            if name!='baseline':self.assertEqual({r['primitive'] for r in rows},{'advance','pause','retreat','roll_differential'})

    def test_legacy_missing_load_not_valid_training_target(self):
        root=Path(__file__).resolve().parents[1]
        sample=root/'artifacts/m11/20261117T144712_trajectory02/samples.jsonl'
        if not sample.exists():self.skipTest('Local raw baseline not distributed in lightweight code checkout')
        for line in sample.read_text().splitlines()[20:25]:
            row=json.loads(line)
            self.assertIsNone(extract(row))

    def test_factorial_parameters_are_independent(self):
        root=Path(__file__).resolve().parents[1]
        plan=json.loads((root/'dynadress/configs/m2_collection/plan.json').read_text())['episodes']
        self.assertEqual(len(plan),100)
        combinations={(i['K'],i['D'],i['friction']) for i in plan}
        self.assertEqual(len(combinations),5*2*5)
        for k in {i['K'] for i in plan}:
            self.assertEqual(sum(i['K']==k for i in plan),20)


if __name__=='__main__':unittest.main()
