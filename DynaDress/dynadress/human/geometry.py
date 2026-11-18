"""Measured passive-arm geometry, in world metres; no Isaac dependency.

Order is shoulder X/Y/Z (intrinsic XYZ), then elbow local Y. Positive forearm
axis points from wrist toward elbow, matching m2_collect_episode.py exactly.
"""
import numpy as np


def _rotation(axis, a):
    a = np.asarray(a); c, s = np.cos(a), np.sin(a)
    r = np.broadcast_to(np.eye(3), a.shape + (3, 3)).copy()
    i, j = ((1, 2), (2, 0), (0, 1))[axis]
    r[..., i, i] = c; r[..., j, j] = c
    r[..., i, j] = -s; r[..., j, i] = s
    return r


def arm_landmarks(q, shoulder=(1.1, 0., 1.8), upper_length=.30, forearm_length=.26):
    q = np.asarray(q)
    if q.shape[-1] != 4: raise ValueError('Expected shoulder_x/y/z, elbow_y')
    upper = _rotation(0, q[..., 0]) @ _rotation(1, q[..., 1]) @ _rotation(2, q[..., 2])
    forearm = upper @ _rotation(1, q[..., 3])
    axis = forearm[..., :, 2]
    elbow = np.asarray(shoulder) - upper_length * upper[..., :, 2]
    wrist = elbow - forearm_length * axis
    return dict(wrist=wrist, forearm_axis=axis, elbow=elbow, hand_tip=elbow-.385*axis)


def axial_progress(q, cuff_center, **kwargs):
    f = arm_landmarks(q, **kwargs)
    return np.sum((np.asarray(cuff_center)-f['wrist'])*f['forearm_axis'], axis=-1)
