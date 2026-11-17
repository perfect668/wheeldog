"""URDF FK and bounded damped-least-squares IK for the supplied dual AR5 chains."""
from pathlib import Path
import xml.etree.ElementTree as E
import numpy as np
from scipy.spatial.transform import Rotation as R

class H1Kinematics:
    def __init__(self,urdf):
        tree=E.parse(urdf).getroot();self.joints={j.find('child').get('link'):j for j in tree.findall('joint')}
        self.names=[j.get('name') for j in tree.findall('joint') if j.get('type')=='revolute']
        self.limits={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in self.joints.values() if j.get('type')=='revolute'}
        self.velocity={j.get('name'):float(j.find('limit').get('velocity')) for j in self.joints.values() if j.get('type')=='revolute'}
    def fk(self,q,link):
        chain=[]
        while link in self.joints:
            j=self.joints[link];chain.append(j);link=j.find('parent').get('link')
        tf=np.eye(4)
        for j in reversed(chain):
            origin=j.find('origin');xyz=np.zeros(3) if origin is None else np.fromstring(origin.get('xyz','0 0 0'),sep=' ');rpy=np.zeros(3) if origin is None else np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
            t=np.eye(4);t[:3,:3]=R.from_euler('xyz',rpy).as_matrix();t[:3,3]=xyz;tf=tf@t
            if j.get('type') in ['revolute','continuous','prismatic']:
                axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');value=q.get(j.get('name'),0.)
                t=np.eye(4)
                if j.get('type')=='prismatic':t[:3,3]=axis*value
                else:t[:3,:3]=R.from_rotvec(axis*value).as_matrix()
                tf=tf@t
        return tf
    def jacobian(self,side,q):
        """Spatial geometric Jacobian in the fixed robot base/world frame."""
        chain=[];link='tool_'+side
        while link in self.joints:
            joint=self.joints[link];chain.append(joint);link=joint.find('parent').get('link')
        tf=np.eye(4);axes=[];origins=[]
        for j in reversed(chain):
            origin=j.find('origin');xyz=np.zeros(3) if origin is None else np.fromstring(origin.get('xyz','0 0 0'),sep=' ');rpy=np.zeros(3) if origin is None else np.fromstring(origin.get('rpy','0 0 0'),sep=' ')
            t=np.eye(4);t[:3,:3]=R.from_euler('xyz',rpy).as_matrix();t[:3,3]=xyz;tf=tf@t
            if j.get('type')=='revolute':
                axis=np.fromstring(j.find('axis').get('xyz'),sep=' ');axes.append(tf[:3,:3]@axis);origins.append(tf[:3,3].copy())
                t=np.eye(4);t[:3,:3]=R.from_rotvec(axis*q.get(j.get('name'),0.)).as_matrix();tf=tf@t
        return np.r_[np.array([np.cross(a,tf[:3,3]-o) for a,o in zip(axes,origins)]).T,np.array(axes).T]

    def solve(self,side,target,initial,iterations=100,margin=0.):
        names=[f'AR5-5_08{side}-W4C1C9-ZY2_joint_{i}' for i in range(1,8)];link=f'tool_{side}';q=dict(initial)
        def error(tf):return np.r_[target[:3,3]-tf[:3,3],R.from_matrix(target[:3,:3]@tf[:3,:3].T).as_rotvec()*.2]
        for _ in range(iterations):
            tf=self.fk(q,link);err=error(tf)
            if np.linalg.norm(err)<1e-5:break
            jac=self.jacobian(side,q);jac[3:]*=.2
            dq=jac.T@np.linalg.solve(jac@jac.T+.002**2*np.eye(6),err)
            for n,d in zip(names,np.clip(dq,-.08,.08)):
                lo,hi=self.limits[n];q[n]=float(np.clip(q.get(n,0.)+d,lo+margin,hi-margin))
        return q,{'position_error_m':float(np.linalg.norm(error(self.fk(q,link))[:3])),'rotation_error_rad':float(np.linalg.norm(error(self.fk(q,link))[3:])/.2)}

    def solve_candidates(self,side,target,initial,count=16,seed=0,margin=.15,waypoints=(),min_elbow_outward=None):
        """Deterministic seeds; reject boundary solutions, rank clearance/mobility.

        This is a kinematic ranking. Runtime PhysX still checks actual collisions.
        """
        rng=np.random.default_rng(seed);names=[f'AR5-5_08{side}-W4C1C9-ZY2_joint_{i}' for i in range(1,8)]
        candidates=[]
        for i in range(count):
            start=dict(initial)
            if i:
                for n in names:
                    lo,hi=self.limits[n];start[n]=float(rng.uniform(lo+margin,hi-margin))
            q,error=self.solve(side,target,start,220,margin=margin)
            if error['position_error_m']>.001 or error['rotation_error_rad']>.01:continue
            probe=dict(q);path_valid=True
            for waypoint in waypoints:
                probe,path_error=self.solve(side,waypoint,probe,100,margin=min(margin,.07))
                if path_error['position_error_m']>.001 or path_error['rotation_error_rad']>.01:
                    path_valid=False;break
            if not path_valid:continue
            clearance=min(min(q[n]-self.limits[n][0],self.limits[n][1]-q[n]) for n in names)
            singular=float(np.linalg.svd(self.jacobian(side,q),compute_uv=False)[-1])
            elbow=self.fk(q,f'AR5-5_08{side}-W4C1C9-ZY2_link4')[:3,3]
            outward=(1 if side=='L' else -1)*elbow[1]
            if min_elbow_outward is not None and outward<min_elbow_outward:continue
            score=clearance+singular+max(min(outward,.5),-.5)*.3
            candidates.append((score,q,dict(error,minimum_joint_margin_rad=clearance,min_jacobian_singular=singular,elbow_outward_m=float(outward),seed_index=i)))
        if not candidates:raise ValueError(f'No interior IK solution for {side}')
        _,q,info=max(candidates,key=lambda v:v[0]);return q,info
    def bounded_step(self,current,target,dt,speed=.25):
        result=dict(current)
        for n in self.names:
            delta=np.clip(target[n]-current[n],-min(self.velocity[n],speed)*dt,min(self.velocity[n],speed)*dt)
            result[n]=float(np.clip(current[n]+delta,*self.limits[n]))
        return result
