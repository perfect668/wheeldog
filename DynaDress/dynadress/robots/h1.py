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
    def solve(self,side,target,initial,iterations=100):
        names=[f'AR5-5_08{side}-W4C1C9-ZY2_joint_{i}' for i in range(1,8)];link=f'tool_{side}';q=dict(initial)
        def error(tf):return np.r_[target[:3,3]-tf[:3,3],R.from_matrix(target[:3,:3]@tf[:3,:3].T).as_rotvec()*.2]
        for _ in range(iterations):
            tf=self.fk(q,link);err=error(tf)
            if np.linalg.norm(err)<1e-5:break
            jac=np.empty((6,7))
            for i,n in enumerate(names):
                pert=dict(q);pert[n]=q.get(n,0.)+1e-5;other=self.fk(pert,link)
                jac[:,i]=np.r_[(other[:3,3]-tf[:3,3])/1e-5,R.from_matrix(other[:3,:3]@tf[:3,:3].T).as_rotvec()*.2/1e-5]
            dq=jac.T@np.linalg.solve(jac@jac.T+.002**2*np.eye(6),err)
            for n,d in zip(names,np.clip(dq,-.08,.08)):q[n]=float(np.clip(q.get(n,0.)+d,*self.limits[n]))
        return q,{'position_error_m':float(np.linalg.norm(error(self.fk(q,link))[:3])),'rotation_error_rad':float(np.linalg.norm(error(self.fk(q,link))[3:])/.2)}
    def bounded_step(self,current,target,dt,speed=.25):
        result=dict(current)
        for n in self.names:
            delta=np.clip(target[n]-current[n],-min(self.velocity[n],speed)*dt,min(self.velocity[n],speed)*dt)
            result[n]=float(np.clip(current[n]+delta,*self.limits[n]))
        return result
