"""Guard the episode split, action timing and physical-unit evaluation boundary."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.m21_evaluate import check_partition, joint_limit_checks, mean_property_conditions, metrics, windows

ROOT = Path(__file__).resolve().parents[1]


class EvaluationContracts(unittest.TestCase):
    def test_future_action_is_not_paired_with_current_state(self):
        # State at t=8 is conditioned by past actions [0..7], and candidate
        # action8 predicts state9. This catches a one-step causal misalignment.
        data = dict(state=np.arange(50)[:, None], action=(100 + np.arange(50))[:, None])
        anchors, (history, past, candidates, target) = windows(data, 8, 20)
        self.assertEqual(anchors[0], 8)
        np.testing.assert_array_equal(history[0, :, 0], np.arange(1, 9))
        np.testing.assert_array_equal(past[0, :, 0], np.arange(100, 108))
        self.assertEqual(candidates[0, 0, 0], 108)
        self.assertEqual(target[0, 0, 0], 9)

    def test_property_ablation_preserves_reference_pose(self):
        true = np.arange(26, dtype=float).reshape(2, 13)
        mean = np.arange(13, dtype=float) + 100
        alternatives = mean_property_conditions(true, mean)
        for name, value in alternatives.items():
            np.testing.assert_array_equal(value[:, 9:], true[:, 9:])
        np.testing.assert_array_equal(alternatives['mean_properties'][:, :9], np.repeat(mean[None, :9], 2, 0))
        np.testing.assert_array_equal(alternatives['mean_friction'][:, :8], true[:, :8])

    def test_frozen_parameter_group_mismatch_is_rejected(self):
        folder = ROOT / 'artifacts/m2/dataset'
        if not (folder / 'index.json').exists():
            self.skipTest('Full frozen M2 metadata not installed')
        index = json.loads((folder / 'index.json').read_text())
        frozen = json.loads((folder / 'split_groups.json').read_text())
        self.assertEqual(check_partition(index, frozen, {}), {'train': 70, 'validation': 14, 'test': 16})
        damaged = copy.deepcopy(index)
        damaged[0]['parameter_group'] = next(r['parameter_group'] for r in index if r['split'] == 'test')
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            check_partition(damaged, frozen, {})

    def test_report_uses_N_Nm_and_mm_not_scaled_load(self):
        schema = json.loads((ROOT / 'artifacts/m2/dataset/features.json').read_text())
        error = np.zeros((2, 109))
        error[:, slice(*schema['groups']['cuff_center'])] = .002
        error[:, slice(*schema['groups']['load'])] = np.tile([3., 3., 3., .4, .4, .4], 2)
        result = metrics(error, schema['groups'])
        self.assertEqual(result['cuff_center_rmse_mm'], 2.)
        self.assertEqual(result['load_force_rmse_N'], 3.)
        self.assertAlmostEqual(result['load_torque_rmse_Nm'], .4)

    def test_limit_diagnostic_does_not_modify_predictions(self):
        q = np.zeros((2, 20, 4)); q[:, :, 3] = -1.
        before = q.copy(); result = joint_limit_checks(q, .05)
        self.assertEqual(result['all_steps_component_violations'], 40)
        self.assertEqual(result['endpoints']['1.0']['windows_with_any_violation'], 2)
        self.assertEqual(result['maximum_violation_rad'], 1.)
        np.testing.assert_array_equal(q, before)


if __name__ == '__main__':
    unittest.main()
