"""Train M2.1 on the frozen M2 whole-episode/property split, without force labels."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.m2_train import Windows
from dynadress.models.structured_world_model import StructuredWorldModel


def fit_normalizer(episodes, fs):
    norm = {}
    for key in ['state', 'action', 'condition']:
        values = np.concatenate([d[key] if key != 'condition' else d[key][None]
                                 for d in episodes])
        norm[key + '_mean'] = values.mean(0).tolist()
        norm[key + '_std'] = np.maximum(values.std(0), fs[key + '_std_floor']).tolist()
    loads = np.concatenate([d['state'][:, slice(*fs['groups']['load'])] for d in episodes])
    # Componentwise training-only robust scale. Retain every peak in fitting/loss.
    floors = np.asarray([.5, .5, .5, .05, .05, .05] * 2)
    scale = np.maximum(np.quantile(np.abs(loads), .75, axis=0), floors)
    transformed = np.arcsinh(loads / scale)
    norm.update(load_scale=scale.tolist(), load_mean=transformed.mean(0).tolist(),
                load_std=np.maximum(transformed.std(0), .1).tolist())
    return norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, default=ROOT/'artifacts/m2/dataset')
    p.add_argument('--output', type=Path, default=ROOT/'artifacts/m21/model')
    p.add_argument('--dynamics-config', type=Path, default=None)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=2101)
    p.add_argument('--device', default='cuda')
    p.add_argument('--torque-bound', type=float, default=4.)
    p.add_argument('--learning-rate', type=float, default=.001)
    a = p.parse_args()
    torch.set_num_threads(2)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.device.startswith('cuda'): torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    a.output.mkdir(parents=True, exist_ok=True)
    index = json.loads((a.dataset/'index.json').read_text())
    fs = json.loads((a.dataset/'features.json').read_text())
    counts = {s: sum(r['split'] == s for r in index) for s in ['train', 'validation', 'test']}
    if counts != {'train': 70, 'validation': 14, 'test': 16}:
        raise RuntimeError(f'Full frozen dataset required, got {counts}')
    train = Windows(a.dataset, index, 'train')
    val = Windows(a.dataset, index, 'validation', stride=4)
    norm = fit_normalizer(train.episodes, fs)
    # The training corpus never reaches explicit hold saturation or a joint
    # stop. Use the exact affine substep map in that verified regime. The
    # mechanics module also offers saturation/hard-stop rollout for diagnostics.
    dyn = json.loads(a.dynamics_config.read_text()) if a.dynamics_config else {
        'use_hold_clip': False, 'joint_limit': False, 'substeps': 12}
    kwargs = dict(dynamics_config=dyn, hidden=96, torque_bound=a.torque_bound)
    model = StructuredWorldModel(norm, fs, **kwargs).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=a.batch_size, shuffle=True, num_workers=0)
    validation = DataLoader(val, batch_size=a.batch_size, shuffle=False, num_workers=0)
    groups = fs['groups']
    # Loss terms are means within physical groups: human state cannot disappear
    # inside the much larger robot/cuff feature count.
    group_weights = {'human_q': 2., 'human_dq': .5, 'cuff_center': .5,
                     'cuff_points': .5, 'progress': 1., 'cuff_normal': .1,
                     'cuff_spans': .1, 'cuff_area': .1, 'cuff_planarity': .1,
                     'robot_q': .1, 'robot_dq': .05, 'ee_pose': .1}

    def loss(pred, target, aux):
        terms = {}
        for key, weight in group_weights.items():
            sl = slice(*groups[key])
            e = (pred[..., sl] - target[..., sl]) / model.state_std[sl]
            terms[key] = e.square().mean() * weight
        sl = slice(*groups['load'])
        z_pred = torch.asinh(pred[..., sl] / model.load_scale)
        z_target = torch.asinh(target[..., sl] / model.load_scale)
        terms['wrist_load'] = .25 * torch.nn.functional.huber_loss(
            (z_pred-model.load_mean)/model.load_std,
            (z_target-model.load_mean)/model.load_std, delta=1.)
        terms['torque_regularizer'] = 1e-4*(aux['cloth_torque']/a.torque_bound).square().mean()
        return sum(terms.values()), terms

    config = dict(model='structured_impedance_v1', model_kwargs=kwargs,
                  epochs=a.epochs, patience=a.patience, seed=a.seed,
                  batch_size=a.batch_size, history=8, training_horizon=20,
                  learning_rate=a.learning_rate, group_weights=group_weights,
                  load_loss='Huber(delta=1) on training-scaled asinh, weight .25; no peak removal',
                  human_torque='equivalent generalized garment action; no direct torque labels; tanh bound regularizer',
                  condition_routing='K/D/q_ref -> mechanics only; friction -> learned interaction heads',
                  normalizer=norm, feature_schema=fs,
                  normalization='training episodes only', episode_counts=counts,
                  train_windows=len(train), validation_windows=len(val),
                  split_sha256=hashlib.sha256((a.dataset/'split_groups.json').read_bytes()).hexdigest(),
                  index_sha256=hashlib.sha256((a.dataset/'index.json').read_bytes()).hexdigest(),
                  torch_version=torch.__version__,
                  loss='0.5 next-step + 0.5 open-loop 20-step grouped loss',
                  selection='lowest validation loss only; original reused test/e088 are not pristine held-out evidence',
                  source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in [Path(__file__), ROOT/'dynadress/models/structured_world_model.py',
                                           ROOT/'dynadress/models/impedance.py']})
    for split in counts:
        config[split+'_groups'] = sorted({r['parameter_group'] for r in index if r['split'] == split})
    (a.output/'training_config.json').write_text(json.dumps(config, indent=2))
    start = time.time(); best = float('inf'); history = []; best_epoch = 0
    for epoch in range(a.epochs):
        model.train(); total = 0.; n = 0
        for batch in loader:
            states, past, actions, cond, target = [v.to(a.device) for v in batch]
            optimizer.zero_grad(set_to_none=True)
            pred, aux = model.rollout(states, past, actions, cond, return_aux=True)
            one, _ = loss(pred[:, :1], target[:, :1], {'cloth_torque': aux['cloth_torque'][:, :1]})
            multi, _ = loss(pred, target, aux)
            error = .5*one + .5*multi
            if not torch.isfinite(error): raise RuntimeError('Nonfinite training loss')
            error.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); total += float(error.detach())*len(states); n += len(states)
        model.eval(); vtotal = 0.; vn = 0; parts = {}
        with torch.no_grad():
            for batch in validation:
                states, past, actions, cond, target = [v.to(a.device) for v in batch]
                pred, aux = model.rollout(states, past, actions, cond, return_aux=True)
                one, _ = loss(pred[:, :1], target[:, :1], {'cloth_torque': aux['cloth_torque'][:, :1]})
                multi, components = loss(pred, target, aux)
                vtotal += float(.5*one+.5*multi)*len(states); vn += len(states)
                for k, v in components.items(): parts[k] = parts.get(k, 0.) + float(v)*len(states)
        metric = vtotal/vn
        row = dict(epoch=epoch+1, train_loss=total/n, validation_loss=metric,
                   validation_multistep_terms={k:v/vn for k,v in parts.items()}, elapsed_s=time.time()-start)
        history.append(row)
        (a.output/'learning_curve.json').write_text(json.dumps(history, indent=2))
        if metric < best:
            best = metric; best_epoch = epoch+1
            torch.save(dict(state_dict=model.state_dict(), normalizer=norm,
                            feature_schema=fs, model_kwargs=kwargs, epoch=epoch+1,
                            validation_loss=best, model_version='m21.structured.v1'), a.output/'best.pt')
        print(json.dumps(row), flush=True)
        if epoch+1 >= 31 and epoch+1-best_epoch >= a.patience: break
    print('M21_TRAIN_COMPLETE', json.dumps(dict(epochs=len(history), best_epoch=best_epoch,
          best_validation_loss=best, wall_s=time.time()-start)), flush=True)


if __name__ == '__main__': main()
