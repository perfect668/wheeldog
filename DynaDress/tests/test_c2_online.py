"""Causal observer, bounded candidate proposals and no privileged input contract."""
import ast,json,unittest
from pathlib import Path
import numpy as np
from dynadress.inference.probing import candidate_actions,proposal_constraints,primitive,CANDIDATES,DURATION
from dynadress.inference.segments import phase,summary
from dynadress.inference.belief import JointBelief,SegmentLikelihood,expected_information_gain
ROOT=Path(__file__).resolve().parents[1]


class OnlineContracts(unittest.TestCase):
    def test_candidates_bounded_and_smooth_at_opportunity_edges(self):
        poses={s:np.eye(4) for s in 'LR'}
        for name in CANDIDATES:
            a=candidate_actions(name,poses,np.array([1.,0,0]),np.array([0,1.,0]))
            self.assertEqual(a.shape,(20,30));self.assertTrue(proposal_constraints(a)['feasible'])
            for t in (0.,DURATION):
                self.assertAlmostEqual(primitive(name,t)['speed_scale'],1.)
                self.assertAlmostEqual(primitive(name,t)['roll_rad'],0.)
            # Between control steps, target speed change <= 0.005m/s.
            self.assertLess(np.max(np.linalg.norm(np.diff(a[:,18:21],axis=0),axis=-1)),.0005)

    def test_inference_has_no_truth_or_legacy_template_arguments(self):
        tree=ast.parse((ROOT/'dynadress/inference/online.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef):
                names={x.arg for x in node.args.args}
                self.assertFalse(names&{'cfg','config','condition','episode_id','privileged','K','D','mu','profile'})
        schema=json.loads((ROOT/'artifacts/m2/dataset/features.json').read_text())
        self.assertFalse(any('probe_' in s or 'friction' in s or 'K0' in s for s in schema['state_names']))

    def test_observable_phase_and_baseline_subtraction(self):
        fs=json.loads((ROOT/'artifacts/m2/dataset/features.json').read_text());g=fs['groups']
        s=np.zeros(109);s[:4]=[0,1,0,.35];s[72]=-.20;s[-12:]=3
        future=np.repeat(s[None],5,axis=0);base=np.full(12,3.)
        self.assertEqual(phase(s,future,g,s[:4],base),'precontact')
        future[:,72]=-.05;future[:,1]+=.01
        self.assertEqual(phase(s,future,g,s[:4],base),'sliding')
        np.testing.assert_allclose(summary(s,future,g,base)[-12:],0.)

    def test_separated_predictions_have_positive_expected_information(self):
        b=JointBelief();means=np.zeros((50,21));means[:,0]=b.grid[:,0]
        like=SegmentLikelihood(np.ones(21)*.1)
        score=expected_information_gain(b,means,like,np.random.default_rng(44),samples=256)
        self.assertGreater(score['human_normalized'],.1)
        self.assertAlmostEqual(score['garment_normalized'],0.,places=10)

if __name__=='__main__':unittest.main()
