"""Frozen M2 episode/group evaluation for structured M2.1 dynamics.

No fitting is performed here. Every model sees identical histories and recorded
candidate actions, then rolls out without future measured state. The original
M2 results are copied verbatim; e088 is explicitly a design/regression case.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HORIZON_STEPS = (1, 5, 10, 20)
MODELS = ('persistence', 'pure_prior', 'm2', 'structured', 'mean_properties',
          'mean_K', 'mean_D', 'mean_friction')
LOWER_Q = np.radians([-100., -150., -100., 0.])
UPPER_Q = np.radians([100., 70., 100., 145.])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(error, groups):
    """Component RMSE in original physical units; peaks are never removed."""
    result = {}
    for name, unit, factor in [('human_q', 'rad', 1.), ('human_dq', 'rad_s', 1.),
                               ('cuff_center', 'mm', 1000.), ('progress', 'mm', 1000.),
                               ('cuff_points', 'mm', 1000.)]:
        x = error[..., slice(*groups[name])]
        result[f'{name}_rmse_{unit}'] = float(np.sqrt(np.mean(x * x)) * factor)
    load = error[..., slice(*groups['load'])].reshape(-1, 2, 6)
    result['load_force_rmse_N'] = float(np.sqrt(np.mean(load[..., :3] ** 2)))
    result['load_torque_rmse_Nm'] = float(np.sqrt(np.mean(load[..., 3:] ** 2)))
    return result


def horizon_metrics(error, schema):
    return {str(round(n * schema['dt_s'], 8)): metrics(error[:, n - 1], schema['groups'])
            for n in HORIZON_STEPS if n <= error.shape[1]}


def joint_limit_checks(q, dt):
    """Diagnostic only: never clip an output or exclude its error."""
    violations = (q < LOWER_Q) | (q > UPPER_Q)
    endpoint = {}
    for n in HORIZON_STEPS:
        if n > q.shape[1]:
            continue
        bad = violations[:, n - 1]
        endpoint[str(round(n * dt, 8))] = dict(
            component_violations=int(bad.sum()), windows_with_any_violation=int(bad.any(-1).sum()),
            evaluated_windows=len(q), maximum_absolute_q_rad=float(np.abs(q[:, n - 1]).max()))
    return dict(endpoints=endpoint, all_steps_component_violations=int(violations.sum()),
                all_steps_total_components=int(violations.size),
                maximum_violation_rad=float(np.maximum(np.maximum(LOWER_Q - q, q - UPPER_Q), 0).max()),
                maximum_absolute_q_rad=float(np.abs(q).max()),
                note='Actual USD angular limits; counting predictions does not impose a prediction hard stop')


def windows(data, history, horizon, stride=4):
    # Exactly the original M2 evaluator's anchors and action-time convention.
    anchors = np.arange(history, len(data['state']) - horizon, stride)
    if not len(anchors):
        return anchors, None
    return anchors, tuple(np.stack([data[key][start(t):end(t)] for t in anchors])
                          for key, start, end in [
                              ('state', lambda t: t - history + 1, lambda t: t + 1),
                              ('action', lambda t: t - history, lambda t: t),
                              ('action', lambda t: t, lambda t: t + horizon),
                              ('state', lambda t: t + 1, lambda t: t + horizon + 1)])


def mean_property_conditions(condition, mean):
    """Replace K/D/mu only; configured reference angles remain observable."""
    result = {}
    for name, sl in [('mean_properties', slice(0, 9)), ('mean_K', slice(0, 4)),
                     ('mean_D', slice(4, 8)), ('mean_friction', slice(8, 9))]:
        value = condition.copy()
        value[..., sl] = np.asarray(mean)[sl]
        result[name] = value
    return result


def check_partition(index, frozen, training_config):
    groups = {split: {r['parameter_group'] for r in index if r['split'] == split}
              for split in ('train', 'validation', 'test')}
    counts = {split: sum(r['split'] == split for r in index) for split in groups}
    if counts != {'train': 70, 'validation': 14, 'test': 16}:
        raise ValueError(f'The frozen M2 dataset must contain 70/14/16 episodes: {counts}')
    for row in index:
        if frozen[row['parameter_group']] != row['split']:
            raise ValueError('An episode disagrees with frozen parameter-group split')
    if any(groups[a] & groups[b] for a, b in [('train', 'validation'), ('train', 'test'),
                                            ('validation', 'test')]):
        raise ValueError('Parameter group leakage')
    if 'train_groups' in training_config and set(training_config['train_groups']) != groups['train']:
        raise ValueError('Checkpoint training groups disagree with dataset')
    if 'validation_groups' in training_config and set(training_config['validation_groups']) != groups['validation']:
        raise ValueError('Checkpoint validation groups disagree with dataset')
    return counts


def predictions_for(models, tensor, states, past, actions, condition, mean):
    structured, old = models
    s, p, a, c = map(tensor, (states, past, actions, condition))
    result = {'persistence': np.repeat(states[:, -1:, :], actions.shape[1], axis=1),
              'pure_prior': structured.prior_rollout(s, p, a, c).cpu().numpy(),
              'm2': old.rollout(s, p, a, c).cpu().numpy(),
              'structured': structured.rollout(s, p, a, c).cpu().numpy()}
    for name, alternative in mean_property_conditions(condition, mean).items():
        result[name] = structured.rollout(s, p, a, tensor(alternative)).cpu().numpy()
    return result


def regression_case(record, data, models, tensor, mean, schema, output):
    """Longer forecasts anchored at and after the retained e088 load peak."""
    g = schema['groups']; h = schema['history_samples']
    load = data['state'][:, slice(*g['load'])].reshape(-1, 2, 6)
    peak = int(np.argmax(np.linalg.norm(load[..., :3], axis=-1).max(-1)))
    horizon = min(40, len(data['state']) - peak - 6)
    cases = []
    arrays = {'observed_time': data['time'], 'observed_state': data['state'],
              'condition': data['condition']}
    for label, t in [('at_peak', peak), ('after_peak_0p2s', peak + 4)]:
        states = data['state'][None, t - h + 1:t + 1]
        past = data['action'][None, t - h:t]
        actions = data['action'][None, t:t + horizon]
        truth = data['state'][t + 1:t + horizon + 1]
        condition = data['condition'][None]
        prediction = predictions_for(models, tensor, states, past, actions, condition, mean)
        _, aux = models[0].rollout(tensor(states), tensor(past), tensor(actions), tensor(condition), return_aux=True)
        arrays[label + '_cloth_torque_Nm'] = aux['cloth_torque'].cpu().numpy()[0]
        arrays[label + '_time'] = data['time'][t + 1:t + horizon + 1]
        arrays[label + '_truth'] = truth
        arrays[label + '_actions'] = actions[0]
        detail = {}
        for name in ('persistence', 'pure_prior', 'm2', 'structured'):
            pred = prediction[name][0]
            arrays[label + '_' + name] = pred
            q = pred[:, slice(*g['human_q'])]
            qtrue = truth[:, slice(*g['human_q'])]
            detail[name] = dict(
                human_q_rmse_rad=float(np.sqrt(np.mean((q - qtrue) ** 2))),
                first_second_human_q_rmse_rad=float(np.sqrt(np.mean((q[:20] - qtrue[:20]) ** 2))),
                final_human_q_rmse_rad=float(np.sqrt(np.mean((q[-1] - qtrue[-1]) ** 2))),
                final_reference_distance_rad=float(np.linalg.norm(q[-1] - data['condition'][9:13])),
                maximum_absolute_q_rad=float(np.max(np.abs(q))),
                predicted_joint_limit_violations=int(((q < LOWER_Q) | (q > UPPER_Q)).sum()),
                finite=bool(np.isfinite(pred).all()))
        cases.append(dict(anchor=label, anchor_index=t, anchor_time_s=float(data['time'][t]),
                          horizon_s=horizon * schema['dt_s'],
                          observed_reference_distance_at_anchor_rad=float(np.linalg.norm(
                              data['state'][t, slice(*g['human_q'])] - data['condition'][9:13])),
                          final_observed_reference_distance_rad=float(np.linalg.norm(
                              truth[-1, slice(*g['human_q'])] - data['condition'][9:13])),
                          final_observed_q_rad=truth[-1, slice(*g['human_q'])].tolist(),
                          models=detail))
    np.savez_compressed(output / 'e088_regression_predictions.npz', **arrays)
    result = dict(episode=record['episode_id'], parameter_group=record['parameter_group'],
                  status='Regression/design case; not independent held-out evidence',
                  peak_time_s=float(data['time'][peak]),
                  peak_force_N=np.linalg.norm(load[peak, :, :3], axis=-1).tolist(),
                  cases=cases,
                  note='Long open-loop forecasts use the same recorded actions; no future state is assimilated. '
                       'Joint-limit counts diagnose predictions; outputs are not clipped for evaluation.')
    (output / 'e088_regression.json').write_text(json.dumps(result, indent=2))
    return result


def property_interventions(record, data, models, tensor, mean, schema, output):
    """Identical histories/actions, one property changed; no causal truth claim."""
    g = schema['groups']; h = schema['history_samples']; horizon = 20
    # One ordinary contact window and one anomalous-load recovery window.
    load = data['state'][:, slice(*g['load'])].reshape(-1, 2, 6)
    peak = int(np.argmax(np.linalg.norm(load[..., :3], axis=-1).max(-1)))
    anchors = [int(np.argmin(np.abs(data['time'] - 6.))), peak]
    changes = [('K_low', slice(0, 4), 4.), ('K_high', slice(0, 4), 20.),
               ('D_low', slice(4, 8), .5), ('D_high', slice(4, 8), 2.),
               ('friction_low', slice(8, 9), .1), ('friction_high', slice(8, 9), .8)]
    rows = []; arrays = {}
    for ci, t in enumerate(anchors):
        states = tensor(data['state'][None, t - h + 1:t + 1]); past = tensor(data['action'][None, t - h:t])
        actions = tensor(data['action'][None, t:t + horizon]); c = data['condition'][None].copy()
        nominal = models[0].rollout(states, past, actions, tensor(c)).cpu().numpy()[0]
        prefix = f'case{ci}'; arrays[prefix + '_nominal'] = nominal
        arrays[prefix + '_time'] = data['time'][t + 1:t + horizon + 1]
        arrays[prefix + '_truth'] = data['state'][t + 1:t + horizon + 1]
        for name, sl, value in changes:
            alternative = c.copy(); alternative[:, sl] = value
            pred, aux = models[0].rollout(states, past, actions, tensor(alternative), return_aux=True)
            pred = pred.cpu().numpy()[0]; arrays[prefix + '_' + name] = pred
            delta = pred - nominal
            rows.append(dict(case=ci, anchor_time_s=float(data['time'][t]), changed_property=name,
                             alternative_attributes=alternative[0].tolist(),
                             one_second_difference_from_nominal=metrics(delta[-1], g),
                             final_q_rad=pred[-1, slice(*g['human_q'])].tolist(),
                             final_dq_rad_s=pred[-1, slice(*g['human_dq'])].tolist(),
                             final_progress_m=float(pred[-1, g['progress'][0]]),
                             final_force_norm_N=np.linalg.norm(pred[-1, slice(*g['load'])].reshape(2, 6)[:, :3], axis=-1).tolist(),
                             final_equivalent_cloth_torque_Nm=aux['cloth_torque'].cpu().numpy()[0, -1].tolist()))
    np.savez_compressed(output / 'property_interventions.npz', **arrays)
    result = dict(episode=record['episode_id'], true_attributes=data['condition'].tolist(),
                  interpretation='Model sensitivity with fixed observed history and future actions. '
                                 'Counterfactual PhysX outcomes are not supplied; this does not prove causal identification.',
                  results=rows)
    (output / 'property_interventions.json').write_text(json.dumps(result, indent=2))
    return result


def evaluate_directed(folder, output, models, tensor, mean, schema, batch_size):
    """New physical executions are scored separately from the frozen test set."""
    from dynadress.models.features import extract, attributes
    pooled = {name: [] for name in MODELS}; per_episode = []; source = []
    for sample_file in sorted(folder.rglob('samples.jsonl')):
        run = sample_file.parent
        config = json.loads((run / 'config.json').read_text())
        valid = []
        for line in sample_file.read_text().splitlines():
            row = json.loads(line); item = extract(row)
            if item is not None:
                valid.append((row, item))
        data = dict(state=np.stack([v[0] for _, v in valid]),
                    action=np.stack([v[1] for _, v in valid]), condition=attributes(config),
                    time=np.array([r['simulation_time_s'] for r, _ in valid]))
        anchors, batch = windows(data, schema['history_samples'], 20)
        if batch is None:
            raise ValueError(f'No directed evaluation windows: {run}')
        states, past, actions, truth = batch
        condition = np.repeat(data['condition'][None], len(anchors), 0)
        errors = {name: [] for name in MODELS}
        for start in range(0, len(anchors), batch_size):
            end = start + batch_size
            pred = predictions_for(models, tensor, states[start:end], past[start:end],
                                   actions[start:end], condition[start:end], mean)
            for name, value in pred.items():
                if not np.isfinite(value).all():
                    raise FloatingPointError(f'Nonfinite directed prediction {name} {run}')
                errors[name].append(value - truth[start:end])
        errors = {name: np.concatenate(value) for name, value in errors.items()}
        for name in MODELS:
            pooled[name].append(errors[name])
        result_file = run / 'result.json'
        per_episode.append(dict(episode=run.name, windows=len(anchors), attributes=data['condition'].tolist(),
                                result=json.loads(result_file.read_text()) if result_file.exists() else None,
                                metrics={name: horizon_metrics(error, schema) for name, error in errors.items()},
                                structured_joint_limits=joint_limit_checks(
                                    errors['structured'][..., slice(*schema['groups']['human_q'])] +
                                    truth[..., slice(*schema['groups']['human_q'])], schema['dt_s'])))
        source.append(dict(episode=run.name, samples_sha256=sha256(sample_file), config_sha256=sha256(run / 'config.json')))
        print('M21_EVAL_DIRECTED_EPISODE', run.name, len(anchors), flush=True)
    if not per_episode:
        raise ValueError(f'No directed episode samples found in {folder}')
    pooled = {name: np.concatenate(value) for name, value in pooled.items()}
    summary = dict(split='directed e088 regression reruns', evaluated_episodes=len(per_episode),
                   evaluated_parameter_groups=1, windows=len(pooled['structured']), per_episode=per_episode,
                   metrics={name: horizon_metrics(error, schema) for name, error in pooled.items()},
                   all_horizons={name: {str(round((j + 1) * schema['dt_s'], 8)): metrics(error[:, j], schema['groups'])
                                       for j in range(20)} for name, error in pooled.items()},
                   provenance=source,
                   evaluation='New physical executions, same e088 design/regression parameter combination. '
                              'Deterministic condition repeats and a small alignment perturbation. '
                              'Not an independent parameter-generalization test, not included in frozen M2 scores.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2))
    return summary


def main():
    import torch
    from dynadress.models.world_model import CoupledWorldModel
    from dynadress.models.structured_world_model import load_structured_checkpoint

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/m2/dataset')
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts/m21/model/best.pt')
    p.add_argument('--m2-checkpoint', type=Path, default=ROOT / 'artifacts/m2/model/best.pt')
    p.add_argument('--original-metrics', type=Path, default=ROOT / 'artifacts/m2/evaluation/metrics.json')
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/m21/evaluation')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--stride', type=int, default=4,
                   help='Anchor stride; 4 exactly reproduces M2, 1 provides phase-balanced supplementary endpoints')
    p.add_argument('--directed-episodes', type=Path,
                   help='Optional root containing new diagnostic samples.jsonl; scored separately')
    a = p.parse_args(); torch.set_num_threads(2); a.output.mkdir(parents=True, exist_ok=True)
    if a.stride < 1:
        raise ValueError('Anchor stride must be positive')
    model, checkpoint = load_structured_checkpoint(a.checkpoint, device=a.device); model.eval()
    old_checkpoint = torch.load(a.m2_checkpoint, map_location=a.device, weights_only=False)
    old = CoupledWorldModel(old_checkpoint['normalizer'], old_checkpoint['feature_schema'],
                            old_checkpoint['hidden'], old_checkpoint.get('transport_prior', False)).to(a.device)
    old.load_state_dict(old_checkpoint['state_dict']); old.eval()
    fs = checkpoint['feature_schema']; h = fs['history_samples']; horizon = 20
    if fs != old_checkpoint['feature_schema']:
        raise ValueError('M2 and M2.1 feature schemas differ')
    index = json.loads((a.dataset / 'index.json').read_text())
    config_path = a.checkpoint.parent / 'training_config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else checkpoint
    frozen = json.loads((a.dataset / 'split_groups.json').read_text())
    counts = check_partition(index, frozen, config)
    if config.get('split_sha256', sha256(a.dataset / 'split_groups.json')) != sha256(a.dataset / 'split_groups.json'):
        raise ValueError('Frozen split checksum mismatch')
    if config.get('index_sha256', sha256(a.dataset / 'index.json')) != sha256(a.dataset / 'index.json'):
        raise ValueError('Dataset index checksum mismatch')
    mean = np.asarray(checkpoint['normalizer']['condition_mean'])
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=a.device)
    records = [r for r in index if r['split'] == a.split]
    pooled = {name: [] for name in MODELS}; by_group = {}; per_episode = []; retained = {}
    predicted_q = {name: [] for name in MODELS}; actual_q = []
    target_load_norm = []
    with torch.no_grad():
        for record in records:
            data = dict(np.load(a.dataset / record['file'])); anchors, batch = windows(data, h, horizon, a.stride)
            if batch is None:
                raise ValueError(f'Episode has no full evaluation window: {record["episode_id"]}')
            states, past, actions, truth = batch
            condition = np.repeat(data['condition'][None], len(anchors), 0)
            errors = {name: [] for name in MODELS}
            for start in range(0, len(anchors), a.batch_size):
                end = start + a.batch_size
                prediction = predictions_for((model, old), tensor, states[start:end], past[start:end],
                                             actions[start:end], condition[start:end], mean)
                for name, value in prediction.items():
                    if not np.isfinite(value).all():
                        raise FloatingPointError(f'{name} produced nonfinite prediction in {record["episode_id"]}')
                    errors[name].append(value - truth[start:end])
            errors = {name: np.concatenate(value) for name, value in errors.items()}
            group = by_group.setdefault(record['parameter_group'], {name: [] for name in MODELS})
            for name in MODELS:
                pooled[name].append(errors[name]); group[name].append(errors[name])
                predicted_q[name].append(errors[name][..., slice(*fs['groups']['human_q'])] +
                                         truth[..., slice(*fs['groups']['human_q'])])
            actual_q.append(truth[..., slice(*fs['groups']['human_q'])])
            target_load_norm.append(np.linalg.norm(
                truth[..., slice(*fs['groups']['load'])].reshape(-1, horizon, 2, 6)[..., :3], axis=-1))
            per_episode.append(dict(episode_id=record['episode_id'], parameter_group=record['parameter_group'],
                                    success=record['success'], windows=len(anchors), K=record['K'], D=record['D'],
                                    friction=record['friction'],
                                    metrics={name: horizon_metrics(error, fs) for name, error in errors.items()}))
            if record['episode_id'].startswith('e088_'):
                retained = dict(record=record, data=data)
            print('M21_EVAL_EPISODE', record['episode_id'], len(anchors), flush=True)
        pooled = {name: np.concatenate(value) for name, value in pooled.items()}
        result = dict(
            split=a.split, split_episode_counts=counts, evaluated_episodes=len(records),
            anchor_stride=a.stride,
            evaluated_parameter_groups=len(by_group), windows=len(pooled['structured']),
            checkpoint_epoch=checkpoint.get('epoch'), checkpoint_validation_loss=checkpoint.get('validation_loss'),
            horizons_s=[n * fs['dt_s'] for n in HORIZON_STEPS],
            metrics={name: horizon_metrics(error, fs) for name, error in pooled.items()},
            all_horizons={name: {str(round((j + 1) * fs['dt_s'], 8)): metrics(error[:, j], fs['groups'])
                                 for j in range(horizon)} for name, error in pooled.items()},
            per_episode=per_episode,
            joint_limit_diagnostics={name: joint_limit_checks(np.concatenate(value), fs['dt_s'])
                                     for name, value in predicted_q.items()},
            observed_joint_limit_diagnostics=joint_limit_checks(np.concatenate(actual_q), fs['dt_s']),
            per_parameter_group={group: {name: horizon_metrics(np.concatenate(value), fs)
                                         for name, value in error.items()} for group, error in by_group.items()},
            train_mean_properties=mean[:9].tolist(),
            evaluation='Frozen M2 episode and parameter-group split; known attributes; recorded candidate actions. '
                       'All overlapping windows retained, including failures and load spikes. No future measured state in rollout. '
                       'This reused test set informed M2.1 design; comparisons are continuity/regression evidence.',
            baselines=dict(persistence='All observed states held',
                           pure_prior='Zero learned garment torque, human implicit impedance integration, cuff action translation; wrist load held',
                           m2='Original frozen M2 checkpoint',
                           structured='M2.1 with true known K/D/friction',
                           mean_properties='Only K/D/friction replaced by training means; q_ref retained'),
            units='Loads are net estimated wrist external load in N/Nm, not isolated human contact or measured F/T. '
                  'Predicted human garment torque is equivalent generalized action without direct labels.',
            aggregation='Pooled component RMSE over anchor windows; episode/group scores retained. Overlapping windows are not independent trials.')
        target_load_norm = np.concatenate(target_load_norm)
        result['target_load_exposure'] = {
            str(round((j + 1) * fs['dt_s'], 8)): dict(
                threshold_N=50., target_windows_above_50N=int((target_load_norm[:, j] > 50.).any(-1).sum()),
                target_wrist_samples_above_50N=int((target_load_norm[:, j] > 50.).sum()),
                total_windows=len(target_load_norm), maximum_target_force_norm_N=float(target_load_norm[:, j].max()))
            for j in range(horizon)}
        result['anchor_phase_note'] = (
            'Stride 4 reproduces M2 exactly but a one-frame load spike appears only at certain endpoint phases; '
            'horizon sawtooth is sampling phase, not periodic predicted load. Stride 1 supplementary results sample '
            'all endpoint phases. Neither evaluation removes peaks.')
        result['macro_episode_metrics'] = {
            name: {time: {field: float(np.mean([r['metrics'][name][time][field] for r in per_episode]))
                          for field in result['metrics'][name][time]}
                   for time in result['metrics'][name]} for name in MODELS}
        if retained:
            result['e088_regression'] = regression_case(**retained, models=(model, old), tensor=tensor,
                                                        mean=mean, schema=fs, output=a.output)
            property_interventions(**retained, models=(model, old), tensor=tensor, mean=mean, schema=fs, output=a.output)
        (a.output / 'metrics.json').write_text(json.dumps(result, indent=2))
        if a.directed_episodes is not None:
            evaluate_directed(a.directed_episodes, a.output / 'directed', (model, old), tensor, mean, fs, a.batch_size)
    receipt = dict(dataset_index_sha256=sha256(a.dataset / 'index.json'),
                   split_sha256=sha256(a.dataset / 'split_groups.json'),
                   checkpoint_sha256=sha256(a.checkpoint), m2_checkpoint_sha256=sha256(a.m2_checkpoint),
                   source_sha256={str(Path(__file__).relative_to(ROOT)): sha256(__file__)},
                   test_selection=f'Frozen M2 episodes; anchors range(history, len(episode)-20, {a.stride})',
                   result='All requested evaluation episodes completed')
    if a.original_metrics.exists():
        target = a.output / 'm2_original_metrics.json'; shutil.copy2(a.original_metrics, target)
        receipt['m2_original_metrics_sha256'] = sha256(target)
        original = json.loads(target.read_text())
        if original.get('evaluation_split') == a.split and a.stride == 4:
            differences = []
            for time, fields in original['metrics']['model'].items():
                for key, value in fields.items():
                    equivalent = key.replace('rad/s', 'rad_s')
                    differences.append(abs(value - result['metrics']['m2'][time][equivalent]))
            receipt['m2_recomputed_max_absolute_metric_difference'] = max(differences)
        elif a.stride != 4:
            receipt['m2_original_metric_comparison'] = 'Not numerically compared: supplementary anchor stride differs from original M2 stride4'
    (a.output / 'evaluation_receipt.json').write_text(json.dumps(receipt, indent=2))
    print('M21_EVAL_COMPLETE', json.dumps({name: value['1.0'] for name, value in result['metrics'].items()}), flush=True)


if __name__ == '__main__':
    main()
