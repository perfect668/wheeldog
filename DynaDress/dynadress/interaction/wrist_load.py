"""Native incoming wrist wrench and Newton--Euler subtree compensation.

The raw signal is a solver joint reaction, NOT an isolated garment contact.
Residual is an estimated net external wrench on wrist+tool+fingers, and can
include other contacts, attachment impulses, model error and joint friction.
No human holding torque is used. No force threshold uses this estimate.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def inertial_support(mass, com_world, inertia_world, velocity, acceleration,
                     origin_world, gravity=(0., 0., -9.81)):
    """World wrench required at joint origin in the absence of external loads."""
    force = mass[:, None] * (acceleration[:, :3] - np.asarray(gravity))
    omega = velocity[:, 3:]
    torque = (np.einsum('nij,nj->ni', inertia_world, acceleration[:, 3:])
              + np.cross(omega, np.einsum('nij,nj->ni', inertia_world, omega))
              + np.cross(com_world-origin_world, force))
    return np.r_[force.sum(0), torque.sum(0)]


class WristLoad:
    def __init__(self, robot, stage, indices, to_numpy):
        from pxr import UsdPhysics, Usd
        self.view = robot._articulation_view._physics_view
        self.cpu = to_numpy
        self.indices = indices
        names = robot._articulation_view.body_names
        self.subtrees = {}
        self.joint_frames = {}
        self.masses = to_numpy(self.view.get_masses())[0]
        self.coms = to_numpy(self.view.get_coms())[0, :, :3]
        self.inertias = to_numpy(self.view.get_inertias())[0].reshape(-1, 3, 3)
        for side in 'LR':
            # Fixed palm/tool are merged into wrist7 by the existing importer.
            self.subtrees[side] = [indices[side]] + [i for i,n in enumerate(names) if f'tool_{side}_finger_' in n]
            path = self.view.link_paths[0][indices[side]]
            matches = [UsdPhysics.Joint(p) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/H1'))
                       if p.IsA(UsdPhysics.Joint) and path in [str(t) for t in UsdPhysics.Joint(p).GetBody1Rel().GetTargets()]]
            if len(matches) != 1:
                raise RuntimeError(f'Ambiguous incoming wrist joint: {side}, {len(matches)}')
            j = matches[0]; q = j.GetLocalRot1Attr().Get()
            if len(self.subtrees[side]) != 3:
                raise RuntimeError(f'Expected merged wrist and two fingers: {side}, {names}')
            self.joint_frames[side] = (np.array(j.GetLocalPos1Attr().Get()),
                Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix())
        self.description = dict(
            source='PhysX ArticulationView.get_link_incoming_joint_force',
            raw_order=['Fx_N','Fy_N','Fz_N','Tx_Nm','Ty_Nm','Tz_Nm'],
            raw_frame='incoming joint child frame, origin at localPos1',
            residual_frame='world axes, origin at wrist incoming joint',
            residual_formula='sum Newton-Euler inertial/gravity support minus parent-on-child incoming reaction',
            limitations='net external estimate; not garment-only contact or calibrated physical F/T sensor; includes attachment/other contacts/model error',
            bodies={s:[names[i] for i in ids] for s,ids in self.subtrees.items()},
            mass_kg={s:self.masses[ids].tolist() for s,ids in self.subtrees.items()},
            child_joint_frames={s:dict(position_m=p.tolist(),rotation_matrix=r.tolist()) for s,(p,r) in self.joint_frames.items()})

    def read(self):
        poses=self.cpu(self.view.get_link_transforms())[0]
        velocities=self.cpu(self.view.get_link_velocities())[0]
        accelerations=self.cpu(self.view.get_link_accelerations())[0]
        incoming=self.cpu(self.view.get_link_incoming_joint_force())[0]
        rot=Rotation.from_quat(poses[:,3:]).as_matrix()
        com=poses[:,:3]+np.einsum('nij,nj->ni',rot,self.coms)
        iw=rot@self.inertias@rot.transpose(0,2,1)
        raw={};estimate={};support={};origins={}
        for side in 'LR':
            i=self.indices[side];ids=self.subtrees[side];p,r=self.joint_frames[side]
            origin=poses[i,:3]+rot[i]@p;jrot=rot[i]@r
            reaction=np.r_[jrot@incoming[i,:3],jrot@incoming[i,3:]]
            predicted=inertial_support(self.masses[ids],com[ids],iw[ids],velocities[ids],accelerations[ids],origin)
            raw[side]=incoming[i].copy();support[side]=predicted
            estimate[side]=predicted-reaction;origins[side]=origin
        if not np.isfinite(np.concatenate(list(raw.values())+list(estimate.values()))).all():
            raise FloatingPointError('Nonfinite wrist reaction')
        return dict(raw_child_frame_N_Nm=raw,origin_world_m=origins), dict(
            source='wrist Newton-Euler residual; see metadata.load_channel',
            external_world_N_Nm=estimate,inertial_support_world_N_Nm=support)
