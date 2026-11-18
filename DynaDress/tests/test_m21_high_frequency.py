"""Physical interval aggregation contracts independent of Isaac availability."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from dynadress.recording.high_frequency import HighFrequencyRecorder


class HighFrequencyAggregationTest(unittest.TestCase):
    def test_signed_impulse_peak_at_last_sample_and_partial_bin(self):
        # Two opposing half-intervals cancel in the signed mean even though the
        # average magnitude is nonzero. The final sample carries the true peak.
        wrenches=np.zeros((13,6))
        wrenches[:6,0]=4
        wrenches[6:11,0]=-4
        wrenches[11]=[-16,0,0,0,6,0]
        wrenches[12]=[2,0,0,0,-3,0]
        with tempfile.TemporaryDirectory() as temporary:
            recorder=HighFrequencyRecorder(temporary)
            for i,wrench in enumerate(wrenches):
                recorder.write(dict(physics_step=100+i,control_step=i//12,
                    simulation_time_s=5+i/240,state='advance',
                    solver_state={'raw_child_frame_N_Nm':{'L':wrench,'R':-wrench}},
                    estimated_load={'external_world_N_Nm':{'L':wrench,'R':-wrench}},
                    human={'q_rad':[0]*4,'dq_rad_s':[0]*4}))
            recorder.close()
            rows=[json.loads(line) for line in (Path(temporary)/'control_20hz_aggregates.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),2)
        complete,partial=rows
        self.assertTrue(complete['complete'])
        self.assertEqual(complete['sample_count'],12)
        self.assertAlmostEqual(complete['integrated_duration_s'],.05)
        left=complete['estimated_world']['L'];right=complete['estimated_world']['R']
        np.testing.assert_allclose(left['mean_signed_wrench_N_Nm'],[-1,0,0,0,.5,0])
        np.testing.assert_allclose(left['vector_impulse_Ns_Nms'],[-.05,0,0,0,.025,0])
        np.testing.assert_allclose(right['vector_impulse_Ns_Nms'],[.05,0,0,0,-.025,0])
        self.assertAlmostEqual(left['mean_force_norm_N'],5)
        self.assertAlmostEqual(left['peak_force_norm_N'],16)
        self.assertEqual(left['peak_physics_step'],111)
        self.assertAlmostEqual(left['peak_simulation_s'],5+11/240)
        self.assertAlmostEqual(complete['interval_end_exclusive_s'],5.05)
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['sample_count'],1)
        self.assertAlmostEqual(partial['integrated_duration_s'],1/240)
        np.testing.assert_allclose(partial['estimated_world']['L']['vector_impulse_Ns_Nms'],wrenches[12]/240)


if __name__=='__main__':unittest.main()
