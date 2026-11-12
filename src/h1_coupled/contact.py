"""Bilateral lateral hand compliance and exact augmented LIPM integration."""
from dataclasses import replace
import numpy as np
from scipy.linalg import expm
from recoverability.model import Model


class CoupledModel:
    def __init__(self,human,contact,dt):
        self.physical=Model(**human);self.contact=dict(contact);self.dt=float(dt)
        self.k=float(contact['stiffness_per_hand_n_m']);self.c=float(contact['damping_per_hand_ns_m'])
        self.fz=float(contact['vertical_per_hand_n']);cap=float(contact['resultant_cap_per_hand_n'])
        if self.k<=0 or self.c<0 or self.fz<0 or cap<=self.fz:raise ValueError('invalid contact parameters')
        gravity=self.physical.gravity_m_s2-(2*self.fz/self.physical.mass_kg if contact['vertical_load_correction'] else 0)
        self.human=replace(self.physical,gravity_m_s2=gravity,dt_s=self.dt)
        self.cap_y=float(np.sqrt(cap**2-self.fz**2))
        # Start from the verified M0 LIPM coefficient and add actual hand forces.
        self.M=np.zeros((4,4));self.M[0,1]=1;self.M[1,0]=self.human.omega**2-2*self.k/self.human.mass_kg
        self.M[1,1]=-2*self.c/self.human.mass_kg;self.M[1,2]=2*self.k/self.human.mass_kg
        self.M[1,3]=2*self.c/self.human.mass_kg;self.M[2,3]=1
        self.u=np.array([0.,0,0,1]);self.d=np.array([0.,1/self.human.mass_kg,0,0])
        self.force_row=np.array([-self.k,-self.c,self.k,self.c])
        self._cache={}

    def matrices(self,duration=None):
        t=self.dt if duration is None else float(duration)
        if t not in self._cache:
            aug=np.zeros((6,6));aug[:4,:4]=self.M;aug[:4,4]=self.u;aug[:4,5]=self.d
            e=expm(aug*t);self._cache[t]=(e[:4,:4],e[:4,4],e[:4,5])
        return self._cache[t]

    def step(self,state,acceleration,disturbance=0.,duration=None):
        a,b,d=self.matrices(duration)
        return a@state+b*acceleration+d*disturbance

    def per_hand_force(self,state):return np.asarray(state)@self.force_row

    def energy_power_residual(self,state):
        x,v,y,w=np.asarray(state);f=self.per_hand_force(state)
        # Per hand, human power + robot power + stored energy rate + dissipation.
        return f*(v-w)+self.k*(y-x)*(w-v)+self.c*(w-v)**2
