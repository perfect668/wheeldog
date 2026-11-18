"""Contracts for physical parameter routing and the two distinct load outputs."""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from dynadress.models.structured_world_model import StructuredWorldModel
from scripts.m21_train import fit_normalizer

ROOT = Path(__file__).resolve().parents[1]


class StructuredModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fs = json.loads((ROOT/'artifacts/m2/dataset/features.json').read_text())
        data = dict(np.load(ROOT/'artifacts/m2/dataset/e000_k0_d0_f0_r0.npz'))
        cls.norm = fit_normalizer([data], cls.fs)
        cls.states = torch.from_numpy(data['state'][80:88][None]).float()
        cls.past = torch.from_numpy(data['action'][79:87][None]).float()
        cls.actions = torch.from_numpy(data['action'][87:107][None]).float()
        cls.cond = torch.from_numpy(data['condition'][None]).float()

    def model(self):
        model = StructuredWorldModel(self.norm, self.fs,
            dynamics_config={'use_hold_clip': False, 'joint_limit': False, 'substeps': 12})
        for head in (model.geometry_head, model.human_torque_head, model.wrist_load_head):
            torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
        return model

    def test_geometry_and_wrist_heads_cannot_directly_write_human_state(self):
        m = self.model(); s = self.states.clone(); s[:, -1, 1] -= .04
        expected = m(s, self.past, self.actions[:, 0], self.cond)
        with torch.no_grad():
            m.geometry_head.bias[:8] = 1000.
            m.wrist_load_head.bias[:] = 3.
        actual = m(s, self.past, self.actions[:, 0], self.cond)
        torch.testing.assert_close(actual[:, :8], expected[:, :8])
        self.assertGreater((actual[:, -12:] - expected[:, -12:]).abs().max().item(), 1.)

    def test_known_impedance_changes_unforced_recovery(self):
        m = self.model(); s = self.states.clone()
        s[:, :, :4] = self.cond[:, None, 9:13]; s[:, :, 1] -= .08; s[:, :, 4:8] = 0
        low = self.cond.clone(); high = self.cond.clone()
        low[:, :4] = 4.; high[:, :4] = 20.
        a = m.prior_rollout(s, self.past, self.actions, low)
        b = m.prior_rollout(s, self.past, self.actions, high)
        self.assertLess(abs(b[0, -1, 1]-high[0, 10]), abs(a[0, -1, 1]-low[0, 10]))

    def test_asinh_retains_extreme_load_and_rollout_remains_finite(self):
        m = self.model(); s = self.states.clone(); sl = slice(*self.fs['groups']['load'])
        s[:, -1, sl] = 664.
        encoded = m.encode_state(s)
        s2 = s.clone(); s2[:, -1, sl] = 665.
        self.assertTrue(torch.all(m.encode_state(s2)[:, -1, sl] > encoded[:, -1, sl]))
        pred, aux = m.rollout(s, self.past, self.actions, self.cond, return_aux=True)
        self.assertTrue(torch.isfinite(pred).all())
        self.assertTrue((aux['cloth_torque'].abs() <= m.torque_bound).all())


if __name__ == '__main__': unittest.main()
