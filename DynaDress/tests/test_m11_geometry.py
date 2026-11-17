"""Acceptance geometry must reject near misses and collapsed/oblique openings."""
import unittest
import numpy as np
from dynadress.garments.cuff import contains_section
from dynadress.robots.h1 import H1Kinematics
from pathlib import Path

class GeometryTests(unittest.TestCase):
    def setUp(self):
        phase=np.linspace(0,2*np.pi,80,endpoint=False)
        self.ring=np.c_[np.zeros(80),.066*np.cos(phase),.066*np.sin(phase)]
    def test_section_containment_and_side_sign(self):
        for x in (-.1,.1):
            result=contains_section(self.ring,np.array([x,0,0]),.035,np.array([1,0,0]))
            self.assertTrue(result['inside']);self.assertAlmostEqual(result['plane_distance_m'],x)
        self.assertFalse(contains_section(self.ring,np.array([0,.04,0]),.035,np.array([1,0,0]))['inside'])
    def test_collapsed_and_oblique_aperture_rejected(self):
        ring=self.ring.copy();ring[:,2]*=.1
        self.assertFalse(contains_section(ring,np.zeros(3),.035,np.array([1,0,0]))['inside'])
        self.assertFalse(contains_section(self.ring,np.zeros(3),.035,np.array([.8,.6,0]))['inside'])
    def test_analytic_jacobian_matches_motion(self):
        model=H1Kinematics(Path(__file__).resolve().parents[1]/'assets/h1/generated/h1_task.urdf')
        q={n:.3 for n in model.names};jac=model.jacobian('L',q);base=model.fk(q,'tool_L')[:3,3]
        for col,name in enumerate(n for n in model.names if '08L' in n):
            perturb=dict(q);perturb[name]+=1e-6
            np.testing.assert_allclose(jac[:3,col],(model.fk(perturb,'tool_L')[:3,3]-base)/1e-6,atol=5e-7)

if __name__=='__main__':unittest.main()
