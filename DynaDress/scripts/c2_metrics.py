"""Physical-unit-separated prediction errors (force and moment never pooled)."""
import numpy as np


def error_components(error, groups):
    error=np.asarray(error);load=error[...,slice(*groups['load'])].reshape(-1,2,6)
    return dict(q=error[...,slice(*groups['human_q'])].ravel(),
                dq=error[...,slice(*groups['human_dq'])].ravel(),
                progress=error[...,slice(*groups['progress'])].ravel(),
                force_N=load[...,:3].ravel(),torque_Nm=load[...,3:].ravel())
