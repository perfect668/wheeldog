"""Same observation summary and geometry gate for replay, live and imagined data."""
import numpy as np


def summary(initial, future, groups, baseline):
    """Endpoint q/dq, native axial change, mean signed external wrist wrench.

    `future` excludes the start state. Native model progress recursion remains
    frozen; C3's FK-derived progress is deliberately a separate diagnostic.
    """
    g = lambda n: slice(*groups[n])
    end = future[..., -1, :]
    load = np.mean(future[..., g('load')], axis=-2)-baseline
    return np.concatenate([end[..., g('human_q')], end[..., g('human_dq')],
                           end[..., g('progress')]-initial[..., g('progress')], load], axis=-1)


def phase(initial, future, groups, q_ref, baseline):
    g = lambda n: slice(*groups[n])
    progress = float(future[-1, g('progress')][0])
    motion = np.linalg.norm(future[-1, g('human_q')]-q_ref)
    # Observable geometry plus movement/estimated load, no collision truth label.
    load = np.linalg.norm((future[-1, g('load')]-baseline).reshape(2,6)[:, :3])
    contact = progress > -.125 and (motion > .002 or load > .5)
    if not contact: return 'precontact'
    moved = abs(float(future[-1, g('progress')][0]-initial[g('progress')][0]))
    return 'sliding' if moved >= .0005 else 'contact'
