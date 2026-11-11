"""MuJoCo FK/Jacobians, independent URDF FK, and bounded numerical dual IK.

Only mj_kinematics/mj_comPos/mj_jacBody are called. No dynamics/actuators.
Original inertias remain unaltered in an in-memory URDF stripped of geometry.
"""
import copy
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .assets import transform, vector

PREFIX = 'AR5-5_08{}-W4C1C9-ZY2_'
TIPS = [PREFIX.format(s)+'flan_link' for s in 'LR']

class Robot:
    def __init__(self, asset):
        self.asset = asset
        root = copy.deepcopy(asset.root)
        for link in root.findall('link'):
            for e in list(link):
                if e.tag in ['visual', 'collision']:
                    link.remove(e)
        m = ET.SubElement(root,'mujoco')
        ET.SubElement(m,'compiler', fusestatic='false', discardvisual='true')
        self.adapted_xml = ET.tostring(root, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.adapted_xml)
        self.data = mujoco.MjData(self.model)
        self.names = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,i)
                      for i in range(self.model.njnt)]
        assert self.model.nq == self.model.nv == len(self.names)
        self.index = {n: int(self.model.jnt_qposadr[i]) for i,n in enumerate(self.names)}
        self.dof = {n: int(self.model.jnt_dofadr[i]) for i,n in enumerate(self.names)}
        self.body_ids = {n: mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n)
                         for n in asset.links}
        self.lower = np.full(len(self.names),-np.inf)
        self.upper = np.full(len(self.names),np.inf)
        for n,i in self.index.items():
            j = asset.joints[n]; limit=j.find('limit')
            if j.get('type') == 'revolute':
                self.lower[i] = float(limit.get('lower')); self.upper[i] = float(limit.get('upper'))
        self.active = [self.index[n] for n in self.names if n in [f'{i}-Joint' for i in range(1,5)]
                       or n.startswith('AR5-')]
        self.tip_ids = [self.body_ids[n] for n in TIPS]

    def fk(self, q, base=(0.,0.,0.)):
        q=np.asarray(q,float); base=np.asarray(base,float)
        if q.shape != (len(self.names),) or base.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(base).all():
            raise ValueError('Nonfinite or malformed configuration')
        # base = x[m], y[m], yaw[rad]; z=0 refers to reference root, NOT the floor.
        bid=self.body_ids[self.asset.base]
        self.model.body_pos[bid] = [base[0],base[1],0.]
        self.model.body_quat[bid] = [np.cos(base[2]/2),0.,0.,np.sin(base[2]/2)]
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model,self.data)
        mujoco.mj_comPos(self.model,self.data)
        result={}
        for n,b in self.body_ids.items():
            t=np.eye(4); t[:3,:3]=self.data.xmat[b].reshape(3,3); t[:3,3]=self.data.xpos[b]
            result[n]=t
        return result

    def jacobians(self, q, base=(0.,0.,0.)):
        self.fk(q,base)
        out=[]
        for b in self.tip_ids:
            jp=np.zeros((3,self.model.nv)); jr=np.zeros_like(jp)
            mujoco.mj_jacBody(self.model,self.data,jp,jr,b)
            out.append(np.vstack([jp,jr]))
        return out  # world linear velocity [m/s], world angular velocity [rad/s]

    def independent_fk(self,q,base=(0.,0.,0.)):
        t=np.eye(4); t[:3,:3]=Rotation.from_euler('z',base[2]).as_matrix(); t[:2,3]=base[:2]
        out={self.asset.base:t}
        pending=list(self.asset.joints.values())
        while pending:
            for j in pending[:]:
                parent=j.find('parent').get('link'); child=j.find('child').get('link')
                if parent not in out: continue
                motion=np.eye(4)
                if j.get('type') in ['revolute','continuous']:
                    motion[:3,:3]=Rotation.from_rotvec(vector(j.find('axis'),'xyz')*q[self.index[j.get('name')]]).as_matrix()
                out[child]=out[parent]@transform(j.find('origin'))@motion
                pending.remove(j)
        return out

    def errors(self,q,base,targets):
        frames=self.fk(q,base)
        errors=[]
        for tip,target in zip(TIPS,targets):
            t=frames[tip]
            errors.append(np.r_[t[:3,3]-target[:3,3],
                                Rotation.from_matrix(t[:3,:3]@target[:3,:3].T).as_rotvec()])
        return np.array(errors)

    def ik(self,targets,base,q0,max_nfev=180,orientation_scale_m=0.25):
        q0=np.asarray(q0,float).copy()
        if not np.isfinite(q0).all(): raise ValueError('Nonfinite IK seed')
        lo=self.lower[self.active]; hi=self.upper[self.active]
        def residual(x):
            q=q0.copy(); q[self.active]=x
            e=self.errors(q,base,targets); e[:,3:]*=orientation_scale_m
            return e.ravel()
        opt=least_squares(residual,np.clip(q0[self.active],lo+1e-10,hi-1e-10),
                          bounds=(lo,hi),max_nfev=max_nfev,ftol=1e-10,xtol=1e-10,gtol=1e-10,
                          jac='3-point',method='trf')
        q=q0.copy(); q[self.active]=opt.x
        return q,dict(status=int(opt.status),message=opt.message,nfev=int(opt.nfev),cost=float(opt.cost))

    def differential_check(self,q,base,eps=1e-6):
        analytic=self.jacobians(q,base); fd=[np.zeros_like(j) for j in analytic]
        for n,k in self.index.items():
            plus=q.copy(); minus=q.copy(); plus[k]+=eps; minus[k]-=eps
            a=self.fk(plus,base); b=self.fk(minus,base)
            for side,tip in enumerate(TIPS):
                fd[side][:3,self.dof[n]]=(a[tip][:3,3]-b[tip][:3,3])/(2*eps)
                fd[side][3:,self.dof[n]]=Rotation.from_matrix(a[tip][:3,:3]@b[tip][:3,:3].T).as_rotvec()/(2*eps)
        own=self.independent_fk(q,base); library=self.fk(q,base)
        return dict(epsilon_rad=eps,
            fk_position_max_m=max(float(np.linalg.norm(own[n][:3,3]-library[n][:3,3])) for n in own),
            fk_rotation_max_rad=max(float(Rotation.from_matrix(own[n][:3,:3]@library[n][:3,:3].T).magnitude()) for n in own),
            jacobian_linear_max_abs=max(float(np.max(np.abs(a[:3]-b[:3]))) for a,b in zip(analytic,fd)),
            jacobian_angular_max_abs=max(float(np.max(np.abs(a[3:]-b[3:]))) for a,b in zip(analytic,fd)))
