"""Opt-in, observation-only physical-step diagnostic for native M2 episodes.

The 20 Hz aggregates contain all 12 pre-integration samples per complete
control interval. Impulse is the rectangular integral, not a peak times 50 ms.
The raw child-frame wrench and estimated world-frame wrench remain separate.
"""
import json
import time
from pathlib import Path
import numpy as np
from .episode import serial


class HighFrequencyRecorder:
    def __init__(self, path, physics_hz=240, control_hz=20):
        self.path = Path(path)
        self.dt = 1. / physics_hz
        self.expected_count = physics_hz // control_hz
        self.raw = (self.path / 'physics_240hz.jsonl').open('w')
        self.control = (self.path / 'control_20hz_aggregates.jsonl').open('w')
        self.pending = []
        self.samples = self.bins = 0

    def write(self, row):
        if self.pending and row['control_step'] != self.pending[0]['control_step']:
            self._flush_bin()
        row = dict(row, sample_wall_time_ns=time.time_ns(), sampling_phase='before next physics integration')
        self.raw.write(json.dumps(row, default=serial, allow_nan=False) + '\n')
        self.pending.append(row)
        self.samples += 1

    def _flush_bin(self):
        rows = self.pending
        if not rows:
            return
        record = dict(control_step=rows[0]['control_step'],
            first_physics_step=rows[0]['physics_step'], last_physics_step=rows[-1]['physics_step'],
            first_simulation_s=rows[0]['simulation_time_s'], last_simulation_s=rows[-1]['simulation_time_s'],
            interval_end_exclusive_s=rows[-1]['simulation_time_s'] + self.dt,
            sample_count=len(rows), integrated_duration_s=len(rows)*self.dt,
            complete=len(rows)==self.expected_count, state_start=rows[0]['state'], state_end=rows[-1]['state'])
        for name, group, field in [
            ('raw_child_frame', 'solver_state', 'raw_child_frame_N_Nm'),
            ('estimated_world', 'estimated_load', 'external_world_N_Nm')]:
            record[name] = {}
            for side in 'LR':
                values = np.array([r[group][field][side] for r in rows])
                norms = np.linalg.norm(values[:, :3], axis=1)
                peak = int(norms.argmax())
                record[name][side] = dict(mean_signed_wrench_N_Nm=values.mean(0),
                    vector_impulse_Ns_Nms=values.sum(0)*self.dt,
                    mean_force_norm_N=float(norms.mean()), peak_force_norm_N=float(norms[peak]),
                    peak_wrench_N_Nm=values[peak], peak_simulation_s=rows[peak]['simulation_time_s'],
                    peak_physics_step=rows[peak]['physics_step'])
        record['human_delta_q_rad'] = np.array(rows[-1]['human']['q_rad']) - np.array(rows[0]['human']['q_rad'])
        record['human_delta_dq_rad_s'] = np.array(rows[-1]['human']['dq_rad_s']) - np.array(rows[0]['human']['dq_rad_s'])
        self.control.write(json.dumps(record, default=serial, allow_nan=False) + '\n')
        self.bins += 1
        self.pending = []
        self.raw.flush()
        self.control.flush()

    def close(self):
        if self.raw.closed:
            return
        self._flush_bin()
        self.raw.close()
        self.control.close()
        (self.path / 'high_frequency_metadata.json').write_text(json.dumps(dict(
            schema='m21-native-physical-step-v1', physics_hz=1/self.dt,
            control_hz=1/self.dt/self.expected_count, physical_samples=self.samples, control_bins=self.bins,
            phase='pre-integration; control commands in a bin apply to its following physical steps',
            averaging='12 equally spaced samples per complete 50 ms bin; partial final bin explicitly marked',
            impulse='sum of signed force/torque samples times 1/240 s',
            load='raw incoming solver reaction in child joint frame; separate Newton-Euler compensated net wrist external estimate in world axes',
            unavailable=['isolated cloth-to-human solver torque', 'measured_wrench'],
            human_damping='implicit velocity drive; -D*dq logged as instantaneous expected damping, not measured applied solver torque',
            implementation='observation only; no extra state writes or changed controller frequency'), indent=2))
