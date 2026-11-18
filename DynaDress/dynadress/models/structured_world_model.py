"""M2.1: known human impedance plus learned equivalent garment action.

The four torque outputs are *effective generalized actions*, learned only from
state prediction. They are not supervised contact torques or wrist F/T readings.
K/D enter the mechanical integrator; friction enters the interaction heads.
"""
from pathlib import Path

import torch
from torch import nn

from dynadress.models.impedance import FixedImpedanceDynamics


class StructuredWorldModel(nn.Module):
    def __init__(self, normalizer, feature_schema, dynamics_config=None,
                 hidden=96, torque_bound=4.0):
        super().__init__()
        self.schema = feature_schema
        self.hidden = hidden
        self.torque_bound = float(torque_bound)
        self.dynamics_config = dynamics_config or {}
        self.dynamics = FixedImpedanceDynamics(**self.dynamics_config)
        for name, value in normalizer.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        ns = len(normalizer['state_mean'])
        na = len(normalizer['action_mean'])
        self.history = nn.GRU(ns + na, hidden, batch_first=True)
        # Known mechanical parameters have no direct learned-head shortcut.
        self.interaction = nn.Sequential(nn.Linear(hidden + na + 1, 128), nn.SiLU(),
                                         nn.Linear(128, 128), nn.SiLU())
        self.geometry_head = nn.Linear(128, ns)
        self.human_torque_head = nn.Linear(128, 4)
        self.wrist_load_head = nn.Linear(128, 12)
        for head in [self.geometry_head, self.human_torque_head, self.wrist_load_head]:
            nn.init.normal_(head.weight, std=.001)
            nn.init.zeros_(head.bias)
        g = self.schema['groups']
        mask = torch.ones(ns)
        for name in ['human_q', 'human_dq', 'load']:
            mask[slice(*g[name])] = 0
        self.register_buffer('geometry_mask', mask)

    def encode_state(self, states):
        x = (states - self.state_mean) / self.state_std
        sl = slice(*self.schema['groups']['load'])
        # Monotonic, no input clipping and no discarded peak samples.
        z = torch.asinh(states[..., sl] / self.load_scale)
        x = x.clone()
        x[..., sl] = (z - self.load_mean) / self.load_std
        return x

    def prior(self, current, action, attributes, human=True):
        """Zero-garment-action impedance + commanded cuff translation prior."""
        g = self.schema['groups']
        nxt = current.clone()
        qs, vs = slice(*g['human_q']), slice(*g['human_dq'])
        if human:
            q, dq = self.dynamics(current[..., qs], current[..., vs],
                                  torch.zeros_like(current[..., qs]),
                                  attributes[..., :4], attributes[..., 4:8],
                                  attributes[..., 9:13], self.schema['dt_s'])
            nxt[..., qs], nxt[..., vs] = q, dq
        p, v = slice(*g['robot_q']), slice(*g['robot_dq'])
        nxt[..., p] = current[..., p] + self.schema['dt_s'] * current[..., v]
        delta = action[..., 18:24].reshape(*action.shape[:-1], 2, 3).mean(-2)
        for name, step in [('cuff_center', delta), ('cuff_points', delta.repeat(1, 8))]:
            sl = slice(*g[name]); nxt[..., sl] = current[..., sl] + step
        normal = current[..., slice(*g['cuff_normal'])]
        normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(.1)
        sl = slice(*g['progress'])
        nxt[..., sl] = current[..., sl] + (delta * normal).sum(-1, keepdim=True)
        return nxt

    def forward(self, states, previous_actions, candidate_action, attributes,
                return_aux=False):
        x = self.encode_state(states)
        past = (previous_actions - self.action_mean) / self.action_std
        _, h = self.history(torch.cat([x, past], -1))
        action = (candidate_action - self.action_mean) / self.action_std
        friction = ((attributes[..., 8:9] - self.condition_mean[8:9]) /
                    self.condition_std[8:9])
        shared = self.interaction(torch.cat([h[-1], action, friction], -1))
        # A bounded unknown input cannot cancel the restoring spring without
        # limit. This is a model regularizer, not a measured contact-force limit.
        tau = self.torque_bound * torch.tanh(self.human_torque_head(shared))
        g = self.schema['groups']; current = states[:, -1]
        out = self.prior(current, candidate_action, attributes, human=False)
        out = out + self.geometry_head(shared) * self.state_std * self.geometry_mask
        qs, vs = slice(*g['human_q']), slice(*g['human_dq'])
        q, dq = self.dynamics(current[..., qs], current[..., vs], tau,
                              attributes[..., :4], attributes[..., 4:8],
                              attributes[..., 9:13], self.schema['dt_s'])
        # q/dq have exactly one update path: the mechanical integrator.
        out[..., qs], out[..., vs] = q, dq
        load_z = self.wrist_load_head(shared) * self.load_std + self.load_mean
        # Only an overflow guard in transformed space; ±20 corresponds to
        # >10^8 times the training scale and does not truncate recorded peaks.
        out[..., slice(*g['load'])] = self.load_scale * torch.sinh(load_z.clamp(-20, 20))
        if return_aux:
            return out, {'cloth_torque': tau, 'load_asinh': load_z}
        return out

    def rollout(self, states, previous_actions, candidate_actions, attributes,
                return_aux=False):
        predictions, torques = [], []
        for action in candidate_actions.unbind(1):
            y, aux = self(states, previous_actions, action, attributes, return_aux=True)
            predictions.append(y); torques.append(aux['cloth_torque'])
            states = torch.cat([states[:, 1:], y[:, None]], 1)
            previous_actions = torch.cat([previous_actions[:, 1:], action[:, None]], 1)
        result = torch.stack(predictions, 1)
        return (result, {'cloth_torque': torch.stack(torques, 1)}) if return_aux else result

    def prior_rollout(self, states, previous_actions, candidate_actions, attributes):
        predictions = []; current = states[:, -1]
        for action in candidate_actions.unbind(1):
            current = self.prior(current, action, attributes)
            predictions.append(current)
        return torch.stack(predictions, 1)


def load_structured_checkpoint(path, device='cpu'):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    model = StructuredWorldModel(checkpoint['normalizer'], checkpoint['feature_schema'],
                                 **checkpoint['model_kwargs']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
