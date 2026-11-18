import unittest
import numpy as np
from dynadress.inference.belief import JointBelief, hypotheses, SegmentLikelihood, expected_information_gain, conditions
from dynadress.human.geometry import arm_landmarks, axial_progress


class InferenceContracts(unittest.TestCase):
    def test_fixed_friction_then_joint_marginals_and_correlation(self):
        for grid in (hypotheses(.3), hypotheses()):
            b=JointBelief(grid); self.assertAlmostEqual(b.weights.sum(),1.)
            self.assertEqual(len(grid),10 if len(np.unique(grid[:,2]))==1 else 50)
            self.assertEqual(conditions(grid,[0,1,0,.35]).shape,(len(grid),13))
        b=JointBelief(); b.logw=np.full(50,-100.); b.logw[0]=b.logw[-1]=np.log(.5)
        kd,mu=b.marginal(); self.assertGreater(np.max(abs(b.weights-kd.repeat(5)*np.tile(mu,10))),.2)

    def test_identical_response_has_zero_information_gain(self):
        b=JointBelief(); b.logw=np.log(np.arange(1,51)/np.arange(1,51).sum())
        score=expected_information_gain(b,np.ones((50,21)),SegmentLikelihood(np.ones(21)),np.random.default_rng(4))
        self.assertLess(abs(score['score']),1e-12)

    def test_update_prefers_generating_hypothesis_and_missing_is_skipped(self):
        b=JointBelief(hypotheses(.3)); pred=np.repeat(np.arange(10,dtype=float)[:,None],21,axis=1)
        like=SegmentLikelihood(np.ones(21)*.05); y=pred[6].copy(); y[9:]=np.nan
        b.update(y,pred,like,'sliding'); self.assertEqual(b.weights.argmax(),6)
        old=b.weights.copy();b.update(y,pred,like,'precontact');np.testing.assert_allclose(old,b.weights)
        y[:]=np.nan;old=b.weights.copy();b.update(y,pred,like,'sliding');np.testing.assert_allclose(old,b.weights)

    def test_fk_known_pose_and_progress(self):
        f=arm_landmarks(np.zeros(4));np.testing.assert_allclose(f['wrist'],[1.1,0,1.24])
        self.assertAlmostEqual(axial_progress(np.zeros(4),f['wrist']+.1*f['forearm_axis']),.1)
        from scipy.spatial.transform import Rotation
        q=np.array([.1,1.,-.2,.35]);r=Rotation.from_euler('XYZ',q[:3]).as_matrix()
        f=arm_landmarks(q);np.testing.assert_allclose(f['forearm_axis'],(r@Rotation.from_euler('y',q[3]).as_matrix())[:,2],atol=1e-14)

if __name__=='__main__':unittest.main()
