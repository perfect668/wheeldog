"""USD four-revolute passive arm. All numbers are configurable simulation assumptions."""
from dataclasses import dataclass
import numpy as np

JOINT_NAMES=('shoulder_x','shoulder_y','shoulder_z','elbow_y')

@dataclass(frozen=True)
class Impedance:
    stiffness: tuple=(4.,4.,2.,2.)
    damping: tuple=(1.,1.,.5,.5)
    reference: tuple=(0.,-1.3,0.,.3)
    torque_limit: tuple=(30.,30.,20.,20.)

    def __post_init__(self):
        for name in ('stiffness','damping','reference','torque_limit'):
            value=np.asarray(getattr(self,name),dtype=float)
            if value.shape!=(4,) or not np.isfinite(value).all():raise ValueError(f'Invalid {name}')
            if name in ('stiffness','damping') and np.any(value<0):raise ValueError('Passive coefficients must be nonnegative')
            if name=='torque_limit' and np.any(value<=0):raise ValueError('Effort bounds must be positive')

    def effort(self,q,dq,gravity):
        q,dq,g=(np.asarray(x,dtype=float) for x in (q,dq,gravity))
        if q.shape!=(4,) or dq.shape!=(4,) or g.shape!=(4,): raise ValueError('Expected 4 DOF')
        if not np.isfinite(np.concatenate([q,dq,g])).all(): raise FloatingPointError('Invalid human state')
        raw=g+np.asarray(self.stiffness)*(np.asarray(self.reference)-q)-np.asarray(self.damping)*dq
        applied=np.clip(raw,-np.asarray(self.torque_limit),np.asarray(self.torque_limit))
        return applied,raw


def build(stage,root='/World/PassiveArm',shoulder=(0.,0.,1.1),upper_length_m=.30,forearm_length_m=.26,upper_radius_m=.045,forearm_radius_m=.035,upper_mass_kg=1.8,forearm_hand_mass_kg=1.0):
    if min(upper_mass_kg,forearm_hand_mass_kg,upper_radius_m,forearm_radius_m)<=0 or upper_length_m<=2*upper_radius_m or forearm_length_m<=2*forearm_radius_m:raise ValueError('Invalid body geometry')
    from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
    UsdGeom.Xform.Define(stage,root)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))
    sx,sy,sz=shoulder
    anchor=UsdGeom.Xform.Define(stage,root+'/anchor');anchor.AddTranslateOp().Set(Gf.Vec3d(sx,sy,sz));UsdPhysics.RigidBodyAPI.Apply(anchor.GetPrim());am=UsdPhysics.MassAPI.Apply(anchor.GetPrim());am.CreateMassAttr(.01);am.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-4,1e-4,1e-4))
    fixed=UsdPhysics.FixedJoint.Define(stage,root+'/fixed_torso');fixed.CreateBody1Rel().SetTargets([root+'/anchor']);fixed.CreateLocalPos0Attr(Gf.Vec3f(*shoulder))
    # Two small inertia carriers implement intersecting shoulder axes; not zero-mass links.
    specs=[('sx',(sx,sy,sz),.001,None),('sy',(sx,sy,sz),.001,None),
           ('upper',(sx,sy,sz),upper_mass_kg,(upper_length_m,upper_radius_m)),('forearm',(sx,sy,sz-upper_length_m),forearm_hand_mass_kg,(forearm_length_m,forearm_radius_m))]
    for name,pos,mass,shape in specs:
        path=root+'/'+name; x=UsdGeom.Xform.Define(stage,path); x.AddTranslateOp().Set(Gf.Vec3d(*pos))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim()); m=UsdPhysics.MassAPI.Apply(x.GetPrim());m.CreateMassAttr(mass)
        if shape:
            length,r=shape
            m.CreateCenterOfMassAttr(Gf.Vec3f(0,0,-length/2))
            diag=(mass*(3*r*r+length*length)/12,)*2+(mass*r*r/2,)
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(*diag))
            cap=UsdGeom.Capsule.Define(stage,path+'/collision'); cap.CreateAxisAttr('Z');cap.CreateHeightAttr(length-2*r);cap.CreateRadiusAttr(r);cap.AddTranslateOp().Set(Gf.Vec3d(0,0,-length/2));cap.CreateDisplayColorAttr([Gf.Vec3f(.75,.52,.38)])
            UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
            collision=PhysxSchema.PhysxCollisionAPI.Apply(cap.GetPrim());collision.CreateContactOffsetAttr(.006);collision.CreateRestOffsetAttr(.001)
        else:m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5,1e-5,1e-5))
    limits=[(-100,100),(-150,70),(-100,100),(0,145)]
    bodies=['sx','sy','upper','forearm']
    for i,(name,axis) in enumerate(zip(JOINT_NAMES,['X','Y','Z','Y'])):
        j=UsdPhysics.RevoluteJoint.Define(stage,root+'/'+name)
        if i:j.CreateBody0Rel().SetTargets([root+'/'+bodies[i-1]])
        j.CreateBody1Rel().SetTargets([root+'/'+bodies[i]])
        if i==0:j.CreateBody0Rel().SetTargets([root+'/anchor'])
        elif i==3:j.CreateLocalPos0Attr(Gf.Vec3f(0,0,-upper_length_m))
        j.CreateAxisAttr(axis);j.CreateLowerLimitAttr(limits[i][0]);j.CreateUpperLimitAttr(limits[i][1])
        drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'angular');drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(1.*np.pi/180.);drive.CreateMaxForceAttr(100.);drive.CreateTargetVelocityAttr(0.)
        # Zero position stiffness; damping coefficient set through tensor API after reset.
    # Hand is rigidly part of forearm collision, avoiding an extra DOF.
    hand=UsdGeom.Capsule.Define(stage,root+'/forearm/hand');hand.CreateAxisAttr('Z');hand.CreateHeightAttr(.09);hand.CreateRadiusAttr(.035);hand.AddTranslateOp().Set(Gf.Vec3d(0,0,-forearm_length_m-.045));hand.CreateDisplayColorAttr([Gf.Vec3f(.75,.52,.38)]);UsdPhysics.CollisionAPI.Apply(hand.GetPrim())
    return root
