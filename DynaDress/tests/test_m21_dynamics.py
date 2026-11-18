"""Mechanical contracts: equilibrium, passive energy, coupling, gradients and gravity."""
import unittest
import torch
from dynadress.models.impedance import FixedImpedanceDynamics,REFERENCE_Q,REFERENCE_MASS,gravity_compensation

class M21Dynamics(unittest.TestCase):
 def test_equilibrium_and_known_gravity(self):
  q=torch.tensor([REFERENCE_Q]);z=torch.zeros_like(q);m=FixedImpedanceDynamics()
  a,b=m(q,z,z,torch.full_like(q,20),torch.full_like(q,.5),q)
  torch.testing.assert_close(a,q,rtol=0,atol=1e-7);torch.testing.assert_close(b,z,rtol=0,atol=1e-7)
  torch.testing.assert_close(gravity_compensation(q),torch.tensor([[0,5.9495921,0,1.2443397]]),atol=2e-6,rtol=1e-6)
 def test_unforced_passive_energy_never_grows(self):
  m=FixedImpedanceDynamics(use_hold_clip=False).double();M=m.mass_matrix;qref=torch.tensor(REFERENCE_Q,dtype=torch.float64)
  for k,d in [(4,.5),(20,2),(20,0)]:
   q=qref+torch.tensor([.05,-.02,.03,.05],dtype=torch.float64);v=torch.tensor([.2,-.1,.1,.2],dtype=torch.float64);last=float('inf')
   for _ in range(200):
    energy=.5*v@M@v+.5*k*((q-qref)**2).sum();self.assertLessEqual(energy.item(),last+1e-10);last=energy.item()
    q,v=m(q,v,torch.zeros(4,dtype=q.dtype),k,d,qref)
   self.assertLess(last,.0001)
 def test_generalized_torque_has_differentiable_coupled_effect(self):
  m=FixedImpedanceDynamics(use_hold_clip=False);q=torch.tensor([REFERENCE_Q]);z=torch.zeros_like(q);t=torch.tensor([[0.,.1,0.,0.]],requires_grad=True)
  k=torch.full_like(q,4,requires_grad=True);d=torch.full_like(q,.5,requires_grad=True)
  a,b=m(q,z,t,k,d,q);self.assertGreater(abs(b[0,3].item()),1e-4)
  (a.square().sum()+b.square().sum()).backward()
  for grad in [t.grad,k.grad,d.grad]:self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)
 def test_batch_dynamics_matches_individual(self):
  m=FixedImpedanceDynamics();q=torch.tensor([REFERENCE_Q]*2);q[:,1]-=.02;v=torch.zeros_like(q);t=torch.zeros_like(q);k=torch.tensor([[4.]*4,[20.]*4]);d=torch.tensor([[.5]*4,[2.]*4]);ref=torch.tensor(REFERENCE_Q)
  a,b=m(q,v,t,k,d,ref)
  for i in range(2):
   x,y=m(q[i],v[i],t[i],k[i],d[i],ref);torch.testing.assert_close(a[i],x);torch.testing.assert_close(b[i],y)
 def test_affine_composition_matches_substep_loop(self):
  torch.manual_seed(2);q=torch.tensor([REFERENCE_Q]*5)+torch.randn(5,4)*.02;v=torch.randn_like(q)*.1;t=torch.randn_like(q)*.1
  K=torch.tensor([4.,8.,12.,16.,20.])[:,None].expand(-1,4);D=torch.full_like(q,.5);ref=torch.tensor(REFERENCE_Q)
  fast=FixedImpedanceDynamics(use_hold_clip=False,joint_limit=False);loop=FixedImpedanceDynamics(use_hold_clip=True,joint_limit=False)
  a,b=fast(q,v,t,K,D,ref);x,y=loop(q,v,t,K,D,ref)
  torch.testing.assert_close(a,x,atol=1e-6,rtol=1e-5);torch.testing.assert_close(b,y,atol=1e-6,rtol=1e-5)
 def test_physical_hardstop_prevents_outward_drift(self):
  m=FixedImpedanceDynamics();q=torch.tensor([REFERENCE_Q]);q[0,1]=m.joint_upper[1]-.001;v=torch.zeros_like(q);v[0,1]=2
  q,v=m(q,v,torch.tensor([[0.,50.,0.,0.]]),4.,.5,torch.tensor(REFERENCE_Q))
  self.assertLessEqual(q[0,1].item(),m.joint_upper[1].item());self.assertEqual(v[0,1].item(),0)
 def test_configuration_roundtrip_and_spd_guard(self):
  m=FixedImpedanceDynamics();other=FixedImpedanceDynamics(**m.configuration());torch.testing.assert_close(m.mass_matrix,other.mass_matrix)
  with self.assertRaises(ValueError):FixedImpedanceDynamics(mass_matrix=torch.zeros(4,4))

if __name__=='__main__':unittest.main()
