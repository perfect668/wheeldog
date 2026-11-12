"""Offline spatial IK charts, independent of a fixed temporal endpoint path.

Sampled nonlinear geometry supplies conservative scalar speed/acceleration and
contact-budget envelopes. LP infeasibility is only within these restrictions.
"""
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
from h1_capability.reachability import pose_error,polynomial_roots_in_unit
from h1_kinematics.assets import dump


class SpatialChart:
    def __init__(self,model,checker,record,config,mobile):
        self.model=model;self.checker=checker;self.config=config;self.mobile=mobile
        self.q0=np.array(record['q_rad']);self.b0=np.array(record['base_xy_yaw']);self.initial=model.poses(self.q0,self.b0)
        self.n=len(model.active);self.width=self.n+(3 if mobile else 0)
        self.z0=np.r_[self.q0[model.active],self.b0] if mobile else self.q0[model.active].copy()
        self.profile=dict(config['_profile']);self.rejected=[]

    def unpack(self,z):
        q=self.q0.copy();q[self.model.active]=z[:self.n]
        return q,z[-3:] if self.mobile else self.b0.copy()

    def tangent(self,z):
        q,b=self.unpack(z);j=self.model.task_jacobian(q,b,self.mobile)
        scale=self.model.coordinate_velocity(self.profile,self.mobile)
        if self.config['chart']['inverse_policy']=='uniform':scale=np.ones_like(scale)
        return scale*np.linalg.lstsq(j*scale,np.tile([0,1,0,0,0,0],2),rcond=1e-9)[0]

    def build(self):
        step=self.config['chart']['step_m'];extent=self.config['chart']['extent_m']
        data={0.:self.z0.copy()}
        for sign in [-1,1]:
            z=self.z0.copy()
            for s in np.arange(step,extent+step/2,step)*sign:
                targets=[t.copy() for t in self.initial]
                for t in targets:t[1,3]+=s
                ok=False
                for iteration in range(18):
                    q,b=self.unpack(z);error=pose_error(self.model.poses(q,b),targets)
                    if np.max(abs(error))<1e-10:ok=True;break
                    j=self.model.task_jacobian(q,b,self.mobile)
                    scale=self.model.coordinate_velocity(self.profile,self.mobile)
                    if self.config['chart']['inverse_policy']=='uniform':scale=np.ones_like(scale)
                    dz=scale*np.linalg.lstsq(j*scale,-error,rcond=1e-9)[0]
                    if not np.isfinite(dz).all() or max(abs(dz))>.5:break
                    z+=dz
                q,b=self.unpack(z)
                reason=None
                if not ok:reason='local_ik_residual_or_step'
                elif np.any(q<self.model.robot.lower-1e-9) or np.any(q>self.model.robot.upper+1e-9):reason='joint_position'
                elif not self.checker.check(self.model.robot.fk(q,b),self.config['robot']['clearance_m'])['passed']:reason='sampled_collision'
                if reason:
                    self.rejected.append(dict(y_m=float(s),reason=reason));break
                data[float(s)]=z.copy()
        self.knots=np.array(sorted(data));self.values=np.array([data[s] for s in self.knots])
        self.slopes=np.array([self.tangent(z) for z in self.values]);self.spline=CubicHermiteSpline(self.knots,self.values,self.slopes)
        return self

    def envelope(self,radius,contact,torso_budget):
        lower=max(-radius,float(self.knots[0]));upper=min(radius,float(self.knots[-1]))
        if lower>=0 or upper<=0:raise ValueError('chart lacks both signs')
        knots=np.unique(np.r_[lower,self.knots[(self.knots>lower)&(self.knots<upper)],upper])
        # Exact coordinate extrema for the restricted spatial spline.
        p=self.spline(knots);slopes=self.spline(knots,1);sp=CubicHermiteSpline(knots,p,slopes)
        d1=np.zeros(self.width);d2=np.zeros(self.width);qmin=np.full(self.width,np.inf);qmax=-qmin.copy()
        for i,h in enumerate(np.diff(knots)):
            for j in range(self.width):
                a,b,c,d=sp.c[:,i,j]
                ts=[0,h]+[r*h for r in polynomial_roots_in_unit([3*a*h*h,2*b*h,c])]
                qmin[j]=min(qmin[j],min(np.polyval([a,b,c,d],ts)));qmax[j]=max(qmax[j],max(np.polyval([a,b,c,d],ts)))
                ts=[0,h]+([-b/(3*a)] if abs(a)>1e-14 and 0<-b/(3*a)<h else [])
                d1[j]=max(d1[j],max(abs(np.polyval([3*a,2*b,c],ts))))
                d2[j]=max(d2[j],abs(2*b),abs(6*a*h+2*b))
        amax=self.model.coordinate_acceleration(self.profile,self.mobile);vmax=self.model.coordinate_velocity(self.profile,self.mobile)
        allocation=self.config['chart']['acceleration_allocation']
        V=float(min(np.min(vmax/np.maximum(d1,1e-12)),np.sqrt(np.min((1-allocation)*amax/np.maximum(d2,1e-12)))))
        A=float(np.min(allocation*amax/np.maximum(d1,1e-12)))
        if self.mobile:
            norm1=float(np.hypot(d1[-3],d1[-2]));norm2=float(np.hypot(d2[-3],d2[-2]))
            V=min(V,self.model.config['base_velocity_m_s']/max(norm1,1e-12),np.sqrt((1-allocation)*self.profile['base_acceleration_m_s2']/max(norm2,1e-12)))
            A=min(A,allocation*self.profile['base_acceleration_m_s2']/max(norm1,1e-12))
        V*=self.config['chart']['velocity_safety_factor']
        allowed=float(np.sqrt(contact['resultant_cap_per_hand_n']**2-contact['vertical_per_hand_n']**2))
        flo,fhi=-allowed,allowed;limiting=[];budget=self.model.budgets().copy()
        budget[self.model.torso_indices]=torso_budget
        samples=np.unique(np.concatenate([np.linspace(a,b,3) for a,b in zip(knots[:-1],knots[1:])]))
        pe=re=0.;clearance=np.inf
        for s in samples:
            q,base=self.unpack(self.spline(s));poses=self.model.poses(q,base)
            targets=[t.copy() for t in self.initial]
            for t in targets:t[1,3]+=s
            e=pose_error(poses,targets).reshape(2,6);pe=max(pe,float(max(np.linalg.norm(e[:,:3],axis=1))));re=max(re,float(max(np.linalg.norm(e[:,3:],axis=1))))
            check=self.checker.check(self.model.robot.fk(q,base),self.config['robot']['clearance_m']);clearance=min(clearance,check['min_distance_m'])
            if not check['passed']:raise ValueError('interpolated_chart_collision')
            c=self.model.contact_map(q,base);offset=c@np.tile([0.,0,contact['vertical_per_hand_n']],2);slope=c@np.tile([0.,1,0],2)
            for j in np.flatnonzero(np.isfinite(budget)):
                if abs(slope[j])<1e-10:
                    if abs(offset[j])>budget[j]+1e-8:raise ValueError('baseline_task_torque_infeasible')
                    continue
                lo,hi=sorted([(-budget[j]-offset[j])/slope[j],(budget[j]-offset[j])/slope[j]])
                if lo>flo:flo=float(lo);limiting.append(dict(y_m=float(s),bound='lower',joint=self.model.robot.names[j],force_n=flo))
                if hi<fhi:fhi=float(hi);limiting.append(dict(y_m=float(s),bound='upper',joint=self.model.robot.names[j],force_n=fhi))
        if flo>0 or fhi<0:raise ValueError('baseline_not_admissible')
        if pe>self.config['robot']['position_tolerance_m'] or re>self.config['robot']['rotation_tolerance_rad']:raise ValueError('chart_pose_error')
        if np.any(qmin[:self.n]<self.model.robot.lower[self.model.active]-1e-9) or np.any(qmax[:self.n]>self.model.robot.upper[self.model.active]+1e-9):raise ValueError('interpolated_chart_position')
        return dict(radius_m=radius,y_lower_m=lower,y_upper_m=upper,speed_m_s=V,acceleration_m_s2=A,
            per_hand_force_lower_n=flo,per_hand_force_upper_n=fhi,
            derivative_abs_bound=d1.tolist(),second_derivative_abs_bound=d2.tolist(),
            max_chart_position_error_m=pe,max_chart_rotation_error_rad=re,min_sampled_clearance_m=clearance,
            limiting_torque_updates=limiting,velocity_acceleration_are_sufficient_envelopes=True,
            max_acceleration_coordinate=self.model.robot.names[self.model.active[int(np.argmax(d1[:self.n]/amax[:self.n]))]])

    def save(self,path,envelopes):
        np.savez_compressed(path,knots_m=self.knots,coordinates=self.values,slopes_per_m=self.slopes,
            q0=self.q0,base0=self.b0,mobile=np.array(self.mobile),initial_handle_poses=np.array(self.initial))
        dump(str(path)+'.json',dict(rejected=self.rejected,envelopes=envelopes,
            interpretation='Sampled spatial IK path and sufficient motion envelopes, not a global robot reachable set'))
