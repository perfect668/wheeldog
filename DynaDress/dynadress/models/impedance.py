"""Known four-DOF passive mechanics; learned garment action enters as Nm.

The fixed reference inertia is measured from PhysX, not fitted on the test set.
K and D bypass the network. D appears once, as an implicit velocity drive.
"""
import math
import torch
from torch import nn

JOINT_ORDER=('shoulder_x','shoulder_y','shoulder_z','elbow_y')
REFERENCE_Q=(0.,1.,0.,.35)
REFERENCE_MASS=((.054522328078746796,3.631942746551431e-08,-.006845642812550068,9.378853249586427e-09),
 (3.631942746551431e-08,.24103185534477234,3.110634250180766e-10,.05947510898113251),
 (-.006845642812550068,3.110634250180766e-10,.005048413295298815,1.0139975407182263e-15),
 (9.378853249586427e-09,.05947510898113251,1.0139975407182263e-15,.022839579731225967))
TORQUE_LIMIT=(30.,30.,20.,20.)
JOINT_LOWER=tuple(math.radians(x) for x in (-100.,-150.,-100.,0.))
JOINT_UPPER=tuple(math.radians(x) for x in (100.,70.,100.,145.))


def gravity_compensation(q):
    """Analytical g(q) of passive_arm.py default masses/COMs, in JOINT_ORDER.

    Positive values are the torque required to counter gravity. The tiny
    shoulder carriers have zero lever arm and add no gravity moment.
    """
    x,y,z,e=q.unbind(-1)
    cx,sx,cy,sy,cz,sz,ce,se=x.cos(),x.sin(),y.cos(),y.sin(),z.cos(),z.sin(),e.cos(),e.sin()
    # Columns of R_x R_y R_z.
    r0=torch.stack((cy*cz,cx*sz+sx*sy*cz,sx*sz-cx*sy*cz),-1)
    r1=torch.stack((-cy*sz,cx*cz-sx*sy*sz,sx*cz+cx*sy*sz),-1)
    r2=torch.stack((sy,-sx*cy,cx*cy),-1)
    upper=-.15*r2;elbow=-.30*r2
    fore=elbow-.13*(se[...,None]*r0+ce[...,None]*r2)
    axis0=torch.zeros_like(q[...,:3]);axis0[...,0]=1
    axis1=torch.stack((torch.zeros_like(x),cx,sx),-1)
    axis2=r2;axis3=r1
    # z component of axis x position, multiplied by upward weight.
    def moment(axis,pos,mass):return mass*9.81*(axis[...,0]*pos[...,1]-axis[...,1]*pos[...,0])
    return torch.stack((moment(axis0,upper,1.8)+moment(axis0,fore,1.),
                        moment(axis1,upper,1.8)+moment(axis1,fore,1.),
                        moment(axis2,upper,1.8)+moment(axis2,fore,1.),
                        moment(axis3,fore-elbow,1.)),-1)


class FixedImpedanceDynamics(nn.Module):
    """Backward Euler impedance with a constant learned torque over one control tick.

    forward(q,dq,cloth_torque,K,D,q_ref,dt=.05) returns q_next,dq_next.
    All states/torques have shape (...,4); scalar or per-joint K/D broadcast.
    Torque units Nm, angles rad, mass kg m². No learned gravity or damping.
    `substeps=12` at 20 Hz gives the simulation's 240 Hz mechanical step.

    Explicit hold saturation is included as a residual evaluated at each
    substep's input pose, while K/D use the implicit next state. This first-order
    saturation approximation is exact whenever the simulation hold is unsaturated
    (the verified data/pulse regime). It does not cap external garment torque.
    Physical USD joint limits are projected after each substep and outward velocity
    is zeroed, a perfectly inelastic hardstop approximation. Coriolis, inertia
    variation and the full PhysX joint-impact solve are omitted in this first model.
    """
    def __init__(self,mass_matrix=REFERENCE_MASS,substeps=12,use_hold_clip=True,torque_limit=TORQUE_LIMIT,joint_limit=True):
        super().__init__()
        mass=torch.as_tensor(mass_matrix,dtype=torch.float32)
        if mass.shape!=(4,4) or not torch.isfinite(mass).all() or not torch.allclose(mass,mass.T,atol=1e-7):raise ValueError('Expected finite symmetric 4x4 mass matrix')
        if torch.linalg.eigvalsh(mass).min()<=0:raise ValueError('Mass matrix must be positive definite')
        if not isinstance(substeps,int) or substeps<1:raise ValueError('substeps must be a positive integer')
        self.register_buffer('mass_matrix',mass)
        self.register_buffer('torque_limit',torch.as_tensor(torque_limit,dtype=torch.float32))
        self.register_buffer('joint_lower',torch.tensor(JOINT_LOWER,dtype=torch.float32))
        self.register_buffer('joint_upper',torch.tensor(JOINT_UPPER,dtype=torch.float32))
        self.substeps=substeps;self.use_hold_clip=bool(use_hold_clip);self.joint_limit=bool(joint_limit)

    def configuration(self):
        return {'mass_matrix':self.mass_matrix.detach().cpu().tolist(),'substeps':self.substeps,
                'use_hold_clip':self.use_hold_clip,'torque_limit':self.torque_limit.detach().cpu().tolist(),'joint_limit':self.joint_limit}

    def forward(self,q,dq,cloth_torque,K,D,q_ref,dt=.05):
        if q.shape[-1]!=4 or dq.shape!=q.shape or cloth_torque.shape!=q.shape:raise ValueError('Expected matching (...,4) states and equivalent torques')
        if not isinstance(dt,(float,int)) or not math.isfinite(dt) or dt<=0:raise ValueError('dt must be a positive scalar in seconds')
        K=torch.as_tensor(K,dtype=q.dtype,device=q.device);D=torch.as_tensor(D,dtype=q.dtype,device=q.device);q_ref=torch.as_tensor(q_ref,dtype=q.dtype,device=q.device)
        M=self.mass_matrix.to(q);h=float(dt)/self.substeps
        diag=h*D+h*h*K
        # One factorization per 20 Hz step; constant matrices reused at 240 Hz.
        A=M+torch.diag_embed(torch.broadcast_to(diag,q.shape))
        inverse=torch.linalg.inv(A)
        def mv(matrix,v):return torch.matmul(matrix,v.unsqueeze(-1)).squeeze(-1)
        if not self.use_hold_clip and not self.joint_limit:
            # Exact composition of the same affine mechanical substep. No
            # unsaturated-gravity trig/240 Hz Python autograd graph is needed.
            # The 12-state vector is [q-q_ref, dq, constant cloth torque].
            eye=torch.eye(4,dtype=q.dtype,device=q.device).expand(A.shape)
            zero=torch.zeros_like(A)
            Fv=-h*torch.matmul(inverse,torch.diag_embed(torch.broadcast_to(K,q.shape)))
            Fd=torch.matmul(inverse,M)
            G=h*inverse
            transition=torch.cat((torch.cat((eye+h*Fv,h*Fd,h*G),-1),
                                  torch.cat((Fv,Fd,G),-1),
                                  torch.cat((zero,zero,eye),-1)),-2)
            state=torch.cat((q-q_ref,dq,cloth_torque),-1)
            result=mv(torch.linalg.matrix_power(transition,self.substeps),state)
            return result[...,:4]+q_ref,result[...,4:8]
        for _ in range(self.substeps):
            restoring=K*(q_ref-q)
            if self.use_hold_clip:
                g=gravity_compensation(q)
                raw=g+restoring
                correction=torch.clamp(raw,-self.torque_limit,self.torque_limit)-raw
            else:correction=0.
            dq=mv(inverse,mv(M,dq)+h*(restoring+cloth_torque+correction))
            q=q+h*dq
            if self.joint_limit:
                # Rigid hardstop approximation; no claim to PhysX impact restitution.
                outward=((q<self.joint_lower)&(dq<0))|((q>self.joint_upper)&(dq>0))
                q=torch.clamp(q,self.joint_lower,self.joint_upper)
                dq=torch.where(outward,torch.zeros_like(dq),dq)
        return q,dq
