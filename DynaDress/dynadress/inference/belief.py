"""Joint Bayesian grid and robust segment likelihood. No environment truth API.

Each channel group has a multivariate Student-t with diagonal scale and one
shared radial noise variable. Groups q/dq/relative/load are conditionally
independent; components within a group share heavy-tail fluctuations. No
posterior floors, truth-centred priors or episode/template identifiers are used.
"""
from itertools import product
import numpy as np

GROUPS = {'q': slice(0, 4), 'dq': slice(4, 8), 'relative': slice(8, 9), 'load': slice(9, 21)}


def hypotheses(fixed_mu=None):
    return np.asarray(list(product([4.,8.,12.,16.,20.], [.5,2.],
                         [.1,.2,.3,.5,.8] if fixed_mu is None else [fixed_mu])))


def conditions(grid, q_ref):
    grid = np.asarray(grid)
    return np.c_[np.repeat(grid[:, :1], 4, axis=1), np.repeat(grid[:, 1:2], 4, axis=1),
                 grid[:, 2], np.broadcast_to(q_ref, (len(grid), 4))].astype(np.float32)


def normalize(logw):
    a = np.asarray(logw, dtype=float)
    if not np.isfinite(a).all(): raise FloatingPointError('Nonfinite log belief')
    z = a - a.max(); return z - np.log(np.exp(z).sum())


def entropy(p):
    p = np.asarray(p); return float(-np.sum(p*np.log(np.maximum(p, 1e-300))))


class JointBelief:
    def __init__(self, grid=None):
        self.grid = hypotheses() if grid is None else np.asarray(grid).copy()
        self.logw = np.full(len(self.grid), -np.log(len(self.grid)))
        self.kd = np.unique(self.grid[:, :2], axis=0)
        self.mu = np.unique(self.grid[:, 2])

    @property
    def weights(self): return np.exp(self.logw)

    def marginal(self, weights=None):
        w = self.weights if weights is None else np.asarray(weights)
        kd = np.array([w[np.all(self.grid[:, :2] == v, axis=1)].sum() for v in self.kd])
        mu = np.array([w[self.grid[:, 2] == v].sum() for v in self.mu])
        return kd, mu

    def summary(self):
        kd, mu = self.marginal(); w = self.weights
        classes = np.array([mu[self.mu <= .2].sum(), mu[self.mu == .3].sum(), mu[self.mu >= .5].sum()])
        return dict(joint=w.tolist(), p_KD=kd.tolist(), p_mu=mu.tolist(),
                    effective_sliding_resistance=dict(zip(('low','medium','high'), classes.tolist())),
                    mean=(w@self.grid).tolist(), map=self.grid[w.argmax()].tolist(),
                    entropy_KD=entropy(kd), entropy_mu=entropy(mu))

    def update(self, observation, predicted, likelihood, phase, human_only=False):
        self.logw = normalize(self.logw + likelihood.log_prob(observation, predicted, phase, human_only))
        return self.summary()


class SegmentLikelihood:
    def __init__(self, scales, df=4.):
        self.scale = np.asarray(scales, dtype=float); self.df = float(df)
        if self.scale.shape != (21,) or np.any(self.scale <= 0): raise ValueError('21 positive calibrated scales required')

    def log_prob(self, y, means, phase='sliding', human_only=False):
        means = np.asarray(means); y = np.asarray(y)
        result = np.zeros(means.shape[:-1])
        if phase == 'precontact': return result
        channels = ('q', 'dq') if human_only or phase != 'sliding' else tuple(GROUPS)
        for name in channels:
            sl = GROUPS[name]; residual = (y[..., sl]-means[..., sl])/self.scale[sl]
            valid = np.isfinite(residual)
            # Normalization constants cancel between hypotheses (same calibrated noise).
            count = valid.sum(-1)
            squared = np.sum(np.where(valid, residual, 0.)**2, -1)
            result += -.5*(self.df+count)*np.log1p(squared/self.df)
        return result


def expected_information_gain(belief, predicted, likelihood, rng, samples=96, phase='sliding'):
    """MC mixture samples -> same likelihood -> joint posterior -> marginal entropies.

    Subtract each sample's no-new-information entropy as a control variate.
    Identical predicted distributions therefore yield exactly zero, even for a
    finite Monte Carlo sample. Signed finite-sample estimates are retained.
    """
    predicted = np.asarray(predicted)
    indices = rng.choice(len(belief.grid), samples, p=belief.weights)
    noise = np.zeros((samples, 21))
    for sl in GROUPS.values():
        radial = np.sqrt(rng.chisquare(likelihood.df, samples)/likelihood.df)
        noise[:, sl] = rng.normal(size=(samples, sl.stop-sl.start))/radial[:,None]
    noisy = predicted[indices] + noise*likelihood.scale
    kd0, mu0 = belief.marginal(); gains = []
    for y in noisy:
        post = np.exp(normalize(belief.logw + likelihood.log_prob(y, predicted, phase)))
        kd, mu = belief.marginal(post)
        gains.append([entropy(kd0)-entropy(kd), entropy(mu0)-entropy(mu)])
    gain = np.mean(gains, axis=0)
    denom = np.log([len(belief.kd), max(len(belief.mu), 2)])
    return dict(human_nats=float(gain[0]), garment_nats=float(gain[1]),
                human_normalized=float(gain[0]/denom[0]), garment_normalized=float(gain[1]/denom[1]),
                score=float(np.sum(gain/denom)), samples=samples)
