"""Render physical-unit M2/M2.1 comparisons from completed evaluation artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'persistence': '#888888', 'pure_prior': '#f29e2e', 'm2': '#cb4141',
          'structured': '#1876bd', 'mean_properties': '#9355b3',
          'mean_K': '#3b946e', 'mean_D': '#976939', 'mean_friction': '#b85792'}
LABELS = {'persistence': 'Hold observed state', 'pure_prior': 'Impedance + cuff translation',
          'm2': 'Original M2', 'structured': 'Structured M2.1',
          'mean_properties': 'Mean K, D, friction', 'mean_K': 'Mean K',
          'mean_D': 'Mean D', 'mean_friction': 'Mean friction'}


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / (name + '.png'), dpi=180)
    fig.savefig(output / (name + '.pdf'))
    plt.close(fig)


def horizon_plot(metrics, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fields = [('human_q_rmse_rad', 'Human q RMSE (rad)'),
              ('human_dq_rmse_rad_s', 'Human dq RMSE (rad/s)'),
              ('cuff_center_rmse_mm', 'Cuff center RMSE (mm)'),
              ('progress_rmse_mm', 'Forearm progress RMSE (mm)'),
              ('load_force_rmse_N', 'Net wrist force RMSE (N)'),
              ('load_torque_rmse_Nm', 'Net wrist torque RMSE (Nm)')]
    for ax, (field, label) in zip(axes.flat, fields):
        for name in ('persistence', 'pure_prior', 'm2', 'structured'):
            values = metrics['all_horizons'][name]
            times = sorted(values, key=float)
            ax.plot([float(t) for t in times], [values[t][field] for t in times],
                    color=COLORS[name], label=LABELS[name], lw=1.8)
        ax.set(xlabel='Open-loop horizon (s)', ylabel=label)
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    stride = metrics.get('anchor_stride', 4)
    fig.suptitle(f'Frozen M2 {metrics["split"]} split: {metrics["evaluated_episodes"]} episodes, '
                 f'{metrics["evaluated_parameter_groups"]} combinations, anchor stride {stride}; spikes retained')
    save(fig, output, 'horizon_comparison')


def property_ablation(metrics, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fields = [('human_q_rmse_rad', 'Human q RMSE (rad)'),
              ('human_dq_rmse_rad_s', 'Human dq RMSE (rad/s)'),
              ('cuff_center_rmse_mm', 'Cuff center RMSE (mm)'),
              ('progress_rmse_mm', 'Progress RMSE (mm)'),
              ('load_force_rmse_N', 'Net wrist force RMSE (N)'),
              ('load_torque_rmse_Nm', 'Net wrist torque RMSE (Nm)')]
    names = ('structured', 'mean_properties', 'mean_K', 'mean_D', 'mean_friction')
    for ax, (field, label) in zip(axes.flat, fields):
        vals = [metrics['metrics'][name]['1.0'][field] for name in names]
        ax.bar(np.arange(len(names)), vals, color=[COLORS[n] for n in names])
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(['True', 'All means', 'K mean', 'D mean', 'mu mean'], rotation=25, ha='right')
        ax.set(ylabel=label); ax.grid(axis='y', alpha=.25)
    fig.suptitle('One-second attribute ablation: training means only; configured q_ref retained')
    save(fig, output, 'property_ablation')


def e088_plots(folder, schema):
    source = folder / 'e088_regression_predictions.npz'
    if not source.exists():
        return
    data = dict(np.load(source)); info = json.loads((folder / 'e088_regression.json').read_text())
    g = schema['groups']; observed_time = data['observed_time']; observed = data['observed_state']
    for case in info['cases']:
        label = case['anchor']; times = data[label + '_time']; t0 = case['anchor_time_s']
        selection = (observed_time >= t0 - .5) & (observed_time <= times[-1])
        for detail in (False, True):
            fig, axes = plt.subplots(2, 2, figsize=(11, 7))
            for j, (ax, joint) in enumerate(zip(axes.flat, ['Shoulder X', 'Shoulder Y', 'Shoulder Z', 'Elbow Y'])):
                dim = g['human_q'][0] + j
                ax.plot(observed_time[selection], observed[selection, dim], 'k', lw=2, label='PhysX observed')
                names = ('pure_prior', 'structured') if detail else ('persistence', 'pure_prior', 'm2', 'structured')
                for name in names:
                    ax.plot(times, data[label + '_' + name][:, dim], color=COLORS[name],
                            label=LABELS[name], lw=1.7)
                ax.axvline(t0, color='gray', lw=.8, ls='--')
                ax.axvline(info['peak_time_s'], color='#b32424', lw=.7, ls=':')
                ax.set(xlabel='Simulation time (s)', ylabel=joint + ' (rad)'); ax.grid(alpha=.25)
                ax.legend(fontsize=8)
            fig.suptitle('e088 regression: same history/actions, ' + label.replace('_', ' ') +
                         (' — detailed response' if detail else ' — old/new open-loop recovery'))
            save(fig, folder, 'e088_' + label + ('_detail' if detail else '_comparison'))
    label = info['cases'][0]['anchor']; times = data[label + '_time']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    selection = (observed_time >= info['peak_time_s'] - .5) & (observed_time <= times[-1])
    for side, ax in enumerate(axes[0]):
        obs = observed[:, slice(*g['load'])].reshape(-1, 2, 6)
        ax.plot(observed_time[selection], np.linalg.norm(obs[selection, side, :3], axis=-1),
                'k', lw=1.5, label='20 Hz PhysX-derived load')
        for name in ('m2', 'structured'):
            load = data[label + '_' + name][:, slice(*g['load'])].reshape(-1, 2, 6)
            ax.plot(times, np.linalg.norm(load[:, side, :3], axis=-1), color=COLORS[name], label=LABELS[name])
        ax.set(xlabel='Simulation time (s)', ylabel=('Left' if side == 0 else 'Right') + ' force norm (N)')
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    torque = data[label + '_cloth_torque_Nm']
    for j, name in enumerate(['Shoulder X', 'Shoulder Y', 'Shoulder Z', 'Elbow Y']):
        axes[1, 0].plot(times, torque[:, j], label=name)
    axes[1, 0].set(xlabel='Simulation time (s)', ylabel='Predicted equivalent garment action (Nm)')
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=.25)
    qref = data['condition'][9:13]
    axes[1, 1].plot(observed_time[selection], np.linalg.norm(observed[selection, slice(*g['human_q'])] - qref, axis=-1), 'k', label='PhysX')
    for name in ('m2', 'structured', 'pure_prior'):
        q = data[label + '_' + name][:, slice(*g['human_q'])]
        axes[1, 1].plot(times, np.linalg.norm(q - qref, axis=-1), color=COLORS[name], label=LABELS[name])
    axes[1, 1].set(xlabel='Simulation time (s)', ylabel='Distance from fixed q_ref (rad)')
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=.25)
    fig.suptitle('e088 regression — wrist load and equivalent human action are separate outputs')
    save(fig, folder, 'e088_load_and_equivalent_action')


def intervention_plots(folder, schema):
    source = folder / 'property_interventions.npz'
    if not source.exists():
        return
    data = dict(np.load(source)); info = json.loads((folder / 'property_interventions.json').read_text())
    g = schema['groups']; qref = np.asarray(info['true_attributes'][9:13])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    figload, load_axes = plt.subplots(1, 2, figsize=(11, 4))
    for ci in (0, 1):
        prefix = f'case{ci}'; times = data[prefix + '_time']
        nominal = data[prefix + '_nominal']
        for col, (a, b, quantity, ylabel) in enumerate([
                ('K_low', 'K_high', lambda s: np.linalg.norm(s[:, slice(*g['human_q'])] - qref, axis=-1), 'Distance from q_ref (rad)'),
                ('D_low', 'D_high', lambda s: np.linalg.norm(s[:, slice(*g['human_dq'])], axis=-1), 'Human speed norm (rad/s)'),
                ('friction_low', 'friction_high', lambda s: s[:, g['cuff_spans'][0] + 1] * 1000., 'Cuff minor span (mm)')]):
            ax = axes[ci, col]
            ax.plot(times, quantity(nominal), 'k--', label='True-property prediction')
            for name, color in [(a, '#1b85b7'), (b, '#db6c1a')]:
                ax.plot(times, quantity(data[prefix + '_' + name]), color=color, label=name.replace('_', ' '))
            ax.set(xlabel='Simulation time (s)', ylabel=ylabel); ax.grid(alpha=.25); ax.legend(fontsize=8)
        for name, color in [('nominal', 'k'), ('friction_low', '#1b85b7'), ('friction_high', '#db6c1a')]:
            s = data[prefix + '_' + name]
            force = np.linalg.norm(s[:, slice(*g['load'])].reshape(-1, 2, 6)[:, :, :3], axis=-1).mean(-1)
            load_axes[ci].plot(times, force, color=color, label=name.replace('_', ' '))
        load_axes[ci].set(xlabel='Simulation time (s)', ylabel='Mean bilateral force norm (N)')
        load_axes[ci].grid(alpha=.25); load_axes[ci].legend(fontsize=8)
    fig.suptitle('Fixed-history / fixed-action property interventions; model sensitivity, not counterfactual PhysX truth')
    figload.suptitle('Friction sensitivity of separate wrist-load output; identical history and candidate actions')
    save(fig, folder, 'property_interventions'); save(figload, folder, 'friction_load_intervention')


def table(metrics, output):
    fields = [('human_q_rmse_rad', 'q rad'), ('human_dq_rmse_rad_s', 'dq rad/s'),
              ('cuff_center_rmse_mm', 'cuff mm'), ('progress_rmse_mm', 'progress mm'),
              ('load_force_rmse_N', 'force N'), ('load_torque_rmse_Nm', 'torque Nm')]
    rows = ['| Horizon | Model | ' + ' | '.join(label for _, label in fields) + ' |',
            '| --- | --- | ' + ' | '.join(['---'] * len(fields)) + ' |']
    for time in ['0.25', '0.5', '1.0']:
        for model in ('persistence', 'pure_prior', 'm2', 'structured'):
            values = metrics['metrics'][model][time]
            rows.append('| ' + time + ' s | ' + LABELS[model] + ' | ' +
                        ' | '.join(f'{values[field]:.6g}' for field, _ in fields) + ' |')
    (output / 'comparison_table.md').write_text('\n'.join(rows) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, default=ROOT / 'artifacts/m21/evaluation')
    p.add_argument('--schema', type=Path, default=ROOT / 'artifacts/m2/dataset/features.json')
    p.add_argument('--training', type=Path, default=ROOT / 'artifacts/m21/model/learning_curve.json')
    args = p.parse_args(); folder = args.evaluation
    metrics = json.loads((folder / 'metrics.json').read_text())
    schema = json.loads(args.schema.read_text())
    horizon_plot(metrics, folder); property_ablation(metrics, folder)
    e088_plots(folder, schema); intervention_plots(folder, schema); table(metrics, folder)
    directed_path = folder / 'directed/metrics.json'
    if directed_path.exists():
        directed = json.loads(directed_path.read_text())
        horizon_plot(directed, directed_path.parent)
        property_ablation(directed, directed_path.parent)
        table(directed, directed_path.parent)
    if args.training.exists():
        curve = json.loads(args.training.read_text()); fig, ax = plt.subplots(figsize=(8, 4))
        for name in ('train_loss', 'validation_loss'):
            ax.plot([r['epoch'] for r in curve], [r[name] for r in curve], label=name)
        ax.set(xlabel='Epoch', ylabel='Training-scaled structured objective', yscale='log')
        ax.grid(alpha=.25); ax.legend(); save(fig, folder, 'learning_curve')
    print('M21_PLOTS_COMPLETE', folder)


if __name__ == '__main__':
    main()
