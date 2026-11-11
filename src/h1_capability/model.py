"""Explicit model overlays and point-contact frames on validated M1-B FK/J."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from h1_kinematics.model import Robot,TIPS

DIRECTIONS={'front':np.array([1.,0,0]),'back':np.array([-1.,0,0]),
            'left':np.array([0.,1,0]),'right':np.array([0.,-1,0]),
            'up':np.array([0.,0,1]),'down':np.array([0.,0,-1])}

def skew(x):
    a,b,c=x
    return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])

class CapabilityModel:
    def __init__(self,asset,config):
        self.config=copy.deepcopy(config)
        source=copy.deepcopy(asset)
        for name,axis in config.get('axis_overrides',{}).items():
            axis=np.array(axis,float)
            if axis.shape!=(3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-10:
                raise ValueError('axis override must be unit vector')
            source.joints[name].find('axis').set('xyz',' '.join(map(str,axis)))
        self.robot=Robot(source)
        self.arm_names=[]
        for tip in TIPS:
            # Order derives from the audited root-to-flange chain, not lexical sorting.
            chain=[n for n in source.chain(tip) if n.startswith('AR5-') and source.joints[n].get('type')=='revolute']
            if len(chain)!=7: raise ValueError('Expected 7 serial revolute joints on each arm chain')
            self.arm_names.append(chain)
        self.arm_indices=[[self.robot.index[n] for n in names] for names in self.arm_names]
        self.arm_dofs=[[self.robot.dof[n] for n in names] for names in self.arm_names]
        ranges=np.deg2rad(np.array(config['joint_ranges_deg'],float))
        if ranges.shape!=(7,2) or not np.isfinite(ranges).all() or np.any(ranges[:,0]>=ranges[:,1]):
            raise ValueError('Invalid arm ranges')
        for idx in self.arm_indices:
            self.robot.lower[idx]=ranges[:,0];self.robot.upper[idx]=ranges[:,1]
        self.torso_names=[n for n in source.chain(TIPS[0]) if n in source.chain(TIPS[1]) and source.joints[n].get('type')=='revolute']
        self.torso_indices=[self.robot.index[n] for n in self.torso_names]
        self.active=[i for group in self.arm_indices for i in group]
        self.velocity=np.deg2rad(np.tile(config['velocity_deg_s'],2))
        if config.get('move_torso'):
            if config.get('torso_velocity_rad_s') is None or config.get('torso_acceleration_rad_s2') is None:
                raise ValueError('Moving torso requires explicit assumed velocity and acceleration')
            self.active+=self.torso_indices
            self.velocity=np.r_[self.velocity,np.full(len(self.torso_indices),config['torso_velocity_rad_s'])]
        self.handles=[]
        for side in ['left','right']:
            entry=config['handle_frame'][side];t=np.eye(4)
            t[:3,:3]=Rotation.from_euler('xyz',entry['rpy_rad']).as_matrix();t[:3,3]=entry['xyz_m']
            self.handles.append(t)
        if np.any(self.velocity<=0):raise ValueError('Velocity must be positive')

    def poses(self,q,base):
        frames=self.robot.fk(q,base)
        return [frames[n]@h for n,h in zip(TIPS,self.handles)]

    def jacobians(self,q,base,with_base=False):
        frames=self.robot.fk(q,base);flange_j=self.robot.jacobians(q,base);out=[]
        for n,h,j in zip(TIPS,self.handles,flange_j):
            offset=frames[n][:3,:3]@h[:3,3]
            jp=j[:3]-skew(offset)@j[3:]
            full=np.vstack([jp,j[3:]])
            if with_base:
                p=(frames[n]@h)[:3,3];lever=p-np.r_[base[:2],0.]
                b=np.zeros((6,3));b[0,0]=1;b[1,1]=1;b[:3,2]=np.cross([0,0,1],lever);b[5,2]=1
                full=np.column_stack([full,b])
            out.append(full)
        return out

    def contact_map(self,q,base):
        # f = robot ON human. Reaction on robot = -f. Required actuator
        # compensation tau_contact = +J^T f (no gravity in this budget model).
        jl,jr=self.jacobians(q,base)
        return np.column_stack([jl[:3].T,jr[:3].T])

    def budgets(self,kind='nominal',scale=1.):
        limits=np.full(self.robot.model.nv,np.inf)
        key='nominal_torque_nm' if kind=='nominal' else 'urdf_torque_nm'
        for indices in self.arm_indices: limits[indices]=np.array(self.config[key])*scale
        torso=self.config.get('torso_torque_limits_nm')
        if torso is not None:
            limits[self.torso_indices]=np.array(torso)*scale
        return limits

    def coordinate_velocity(self,profile,mobile):
        return np.r_[self.velocity,[self.config['base_velocity_m_s']]*2+[profile['base_yaw_velocity_rad_s']]] if mobile else self.velocity.copy()

    def coordinate_acceleration(self,profile,mobile):
        a=np.full(len(self.active),profile['joint_acceleration_rad_s2'])
        if self.config.get('move_torso'):a[-len(self.torso_indices):]=self.config['torso_acceleration_rad_s2']
        return np.r_[a,[profile['base_acceleration_m_s2']]*2+[profile['base_yaw_acceleration_rad_s2']]] if mobile else a

    def update(self,q,base,delta,mobile):
        q=q.copy();base=base.copy();q[self.active]+=delta[:len(self.active)]
        if mobile:base+=delta[-3:]  # world x/y and yaw, planar SE(2) chart
        return q,base

    def task_jacobian(self,q,base,mobile):
        columns=self.active+list(range(self.robot.model.nv,self.robot.model.nv+3)) if mobile else self.active
        return np.vstack([j[:,columns] for j in self.jacobians(q,base,mobile)])
