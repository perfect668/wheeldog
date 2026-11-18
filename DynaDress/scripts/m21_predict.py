"""Open-loop candidate-action inference; no future measured state is read."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dynadress.models.structured_world_model import load_structured_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'artifacts/m21/model/best.pt')
    p.add_argument('--sequence', type=Path, required=True)
    p.add_argument('--anchor', type=int, default=160)
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--actions', type=Path, help='Optional horizon x 30 NPY candidate commands')
    p.add_argument('--condition', type=Path, help='Optional 13-vector JSON, same known-property port')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args(); torch.set_num_threads(2)
    model, checkpoint = load_structured_checkpoint(a.checkpoint, a.device)
    fs = checkpoint['feature_schema']; h = fs['history_samples']; g = fs['groups']
    data = dict(np.load(a.sequence)); t = a.anchor
    if a.horizon < 1 or t < h or t >= len(data['state']):
        raise ValueError('Positive horizon and an anchor with full preceding history required')
    actions = np.load(a.actions) if a.actions else data['action'][t:t+a.horizon]
    condition = np.asarray(json.loads(a.condition.read_text()), dtype=np.float32) if a.condition else data['condition']
    if actions.shape != (a.horizon, len(fs['action_names'])):
        raise ValueError('Candidate action shape must be horizon x action_dim')
    if condition.shape != (len(fs['condition_names']),):
        raise ValueError('Expected K[4],D[4],friction,q_ref[4] condition vector')
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32, device=a.device)[None]
    with torch.no_grad():
        pred, aux = model.rollout(tensor(data['state'][t-h+1:t+1]),
            tensor(data['action'][t-h:t]), tensor(actions), tensor(condition), return_aux=True)
    pred = pred.cpu().numpy()[0]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, predicted_state=pred, actions=actions, condition=condition,
        predicted_human_q_rad=pred[:, slice(*g['human_q'])],
        predicted_human_dq_rad_s=pred[:, slice(*g['human_dq'])],
        predicted_cuff_delta_m=pred[:, slice(*g['cuff_center'])]-data['state'][t, slice(*g['cuff_center'])],
        predicted_progress_delta_m=pred[:, g['progress'][0]]-data['state'][t, g['progress'][0]],
        predicted_estimated_wrist_load_N_Nm=pred[:, slice(*g['load'])],
        predicted_equivalent_cloth_torque_Nm=aux['cloth_torque'].cpu().numpy()[0])
    print('M21_PREDICT_COMPLETE', json.dumps(dict(output=str(a.output),
        anchor_time_s=float(data['time'][t]), horizon_s=a.horizon*fs['dt_s'],
        future_state_input=False, action_source='candidate file' if a.actions else 'recorded commands',
        torque_source='learned equivalent generalized action; no direct contact-torque labels',
        wrist_source=fs['load_source'])))


if __name__ == '__main__': main()
