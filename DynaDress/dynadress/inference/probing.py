"""One bounded probing opportunity layered on the original sleeve servo."""
import numpy as np
from scipy.spatial.transform import Rotation
from dynadress.models.features import pose_feature

DURATION=1.0
CANDIDATES=('retreat_readvance','pause_advance','roll_differential','continue')


def primitive(name, elapsed):
    out=dict(primitive='advance',speed_scale=1.,differential_m=0.,roll_rad=0.)
    if name=='continue' or not 0<=elapsed<DURATION:return out
    u=elapsed/DURATION
    if name=='retreat_readvance':
        # Start/end at ordinary advance speed with zero slope. The middle
        # reverses at -0.6x, then re-advances; targets remain continuous.
        out.update(primitive=name,speed_scale=1.-1.60*np.sin(np.pi*u)**2)
    elif name=='pause_advance':
        out.update(primitive=name,speed_scale=float(1.-np.sin(np.pi*u)**2))
    elif name=='roll_differential':
        pulse=np.sin(np.pi*u)**2
        out.update(primitive=name,speed_scale=1.-.35*pulse,differential_m=.0018*pulse,roll_rad=np.radians(2.)*pulse)
    else:raise ValueError(name)
    return out


def candidate_actions(name, commanded, haxis, lateral, speed=.025,dt=.05):
    """Local constant-axis proposal. Actual servo still follows measured geometry.

    These are proposed targets, not claims about future closed-loop commands.
    Actual commands are logged and used for posterior assimilation.
    """
    pose={s:commanded[s].copy() for s in 'LR'};origin={s:v.copy() for s,v in pose.items()}
    translation=np.zeros(3);actions=[]
    for i in range(round(DURATION/dt)):
        probe=primitive(name,i*dt);translation+=speed*dt*probe['speed_scale']*haxis
        previous={s:v.copy() for s,v in pose.items()}
        for s,sign in zip('LR',(1,-1)):
            pose[s]=origin[s].copy();pose[s][:3,3]+=translation+sign*probe['differential_m']*lateral
            pose[s][:3,:3]=Rotation.from_rotvec(probe['roll_rad']*haxis).as_matrix()@origin[s][:3,:3]
        actions.append(np.concatenate([*[pose_feature(pose[s]) for s in 'LR'],
                        *[pose[s][:3,3]-previous[s][:3,3] for s in 'LR'],
                        *[Rotation.from_matrix(pose[s][:3,:3]@previous[s][:3,:3].T).as_rotvec() for s in 'LR']]))
    return np.asarray(actions,dtype=np.float32)


def proposal_constraints(actions):
    delta=actions[:,18:24].reshape(-1,2,3);rot=actions[:,24:30].reshape(-1,2,3)
    return dict(feasible=bool(np.max(np.linalg.norm(delta,axis=-1))<=.0016 and
                              np.max(np.linalg.norm(rot,axis=-1))<=np.radians(.5) and
                              np.max(np.linalg.norm(delta[:,0]-delta[:,1],axis=-1))<=.0006),
                max_translation_step_m=float(np.max(np.linalg.norm(delta,axis=-1))),
                max_rotation_step_rad=float(np.max(np.linalg.norm(rot,axis=-1))))
