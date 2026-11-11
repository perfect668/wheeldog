"""Descriptive population analysis; selection is explicitly outcome-dependent.

No hypothesis test treats overlapping pairs as independent samples. Motion
numbers are achieved path lower bounds, not estimates of global reachable sets.
"""
from pathlib import Path
import argparse
import csv
import json
import subprocess
from collections import Counter
import numpy as np
import yaml
from scipy.stats import spearmanr
from h1_kinematics.assets import dump,sha256
from .model import DIRECTIONS


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pair_metrics(pair, lookup, horizons, gap_m, ratio):
    """Symmetric six-direction comparisons; never choose a single best ray."""
    comparisons = []
    for h in horizons:
        for mobile in [False, True]:
            for direction in DIRECTIONS:
                a = lookup[(pair['a'], h, mobile, direction)]
                b = lookup[(pair['b'], h, mobile, direction)]
                va, vb = a['achieved_displacement_m'], b['achieved_displacement_m']
                valid = va is not None and vb is not None
                low, high = sorted([va, vb]) if valid else (None, None)
                comparisons.append(dict(horizon_s=h, mobile=mobile, direction=direction,
                    a_m=va, b_m=vb, a_status=a['status'], b_status=b['status'],
                    gap_m=abs(va-vb) if valid else None,
                    ratio=high/low if valid and low > 1e-12 else None,
                    low_bound_is_zero=bool(valid and low == 0),
                    practical_difference=bool(valid and high-low >= gap_m and high >= ratio*low)))
    primary = [x for x in comparisons if x['horizon_s'] == .3 and not x['mobile']]
    gaps = [x['gap_m'] for x in primary if x['gap_m'] is not None]
    return dict(**pair, comparisons=comparisons,
        primary_mean_six_direction_gap_m=float(np.mean(gaps)) if len(gaps) == 6 else None,
        primary_practical_directions=sum(x['practical_difference'] for x in primary))


def select_pairs(metrics, count):
    ranked = sorted((p for p in metrics if p['primary_mean_six_direction_gap_m'] is not None),
                    key=lambda p: (-p['primary_mean_six_direction_gap_m'], p['a'], p['b']))
    used = set(); selected = []
    for p in ranked:
        if p['a'] in used or p['b'] in used:
            continue
        selected.append(p); used.update([p['a'], p['b']])
        if len(selected) == count:
            break
    return selected


def describe(values):
    values = [x for x in values if x is not None and np.isfinite(x)]
    if not values:
        return dict(n=0)
    return dict(n=len(values), minimum=float(min(values)), median=float(np.median(values)),
                maximum=float(max(values)), mean=float(np.mean(values)))


def summarize_cases(rows):
    attempts = [a for r in rows for a in r.get('attempts', [])]
    return dict(case_count=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
        attempt_statuses=dict(Counter(a['status'] for a in attempts)),
        rejection_reasons=dict(Counter(reason for a in attempts for reason in a.get('reasons', []))),
        numerical_errors=[dict(case_id=r['case_id'], attempt=a) for r in rows for a in r.get('attempts', [])
                          if a['status'] == 'NUMERICAL_ERROR'],
        displacement_m=describe([r.get('achieved_displacement_m') for r in rows]),
        wall_seconds_per_case=describe([r.get('elapsed_s') for r in rows]),
        cumulative_worker_seconds=sum(r.get('elapsed_s', 0) for r in rows),
        fixed_subset_adoptions=sum(r.get('selected_path_origin') == 'fixed_base_subset' for r in rows),
        max_sampled_position_error_m=max((r.get('best', {}).get('max_position_error_m', 0) for r in rows if r.get('best')), default=0),
        max_sampled_rotation_error_rad=max((r.get('best', {}).get('max_rotation_error_rad', 0) for r in rows if r.get('best')), default=0),
        peak_acceleration_coordinates=dict(Counter(r['best']['max_acceleration_coordinate'] for r in rows if r.get('best'))))


def analyze(output, plots=True):
    out = Path(output); read = lambda p: json.loads((out/p).read_text())
    config = yaml.safe_load((out/'config.yaml').read_text())
    force = read('force_and_instantaneous.json'); reach = read('reachability.json')
    shared_indices = read('model_mapping.json')['shared_torso_indices']
    accepted = [r for r in force if r['status'] == 'ACCEPTED']
    pairs = [r for r in read('all_force_pairs.json') if r['status'] == 'MATCHED']
    lookup = {(r['candidate'], r['horizon_s'], r['mobile'], r['direction']): r for r in reach}
    metrics = [pair_metrics(p, lookup, config['motion']['horizons_s'],
        config['force']['practical_reach_gap_m'], config['force']['practical_reach_ratio']) for p in pairs]
    dump(out/'pair_comparisons.json', metrics)
    selected = select_pairs(metrics, config['force']['selected_pair_count'])
    dump(out/'selected_pairs.json', selected)
    selected_ids={x for p in selected for x in [p['a'],p['b']]}
    dump(out/'typical_configuration_constraints.json',dict(
        usage='Review input candidates for a future stage only; no human recovery experiment performed',
        force_semantics='Arm contact-torque-budget relaxation; unknown shared torso limits, no gravity or base stability',
        motion_semantics='Achieved sampled paths, not a reachable envelope or causal controller guarantee',
        model=config['model'],nominal_assumed_motion=config['profiles']['nominal'],
        task_force=config['force'],
        configurations=[r for r in accepted if r['id'] in selected_ids],
        main_paths=[{k:v for k,v in r.items() if k!='attempts'} for r in reach if r['candidate'] in selected_ids]))
    pair_rows = [dict(a=p['a'], b=p['b'], force_gap=p['max_relative_gap'], **v)
                 for p in metrics for v in p['comparisons']]
    write_csv(out/'pair_comparisons.csv', pair_rows)
    force_rows = []; instant_rows = []
    for r in accepted:
        for budget, kinds in r['force'].items():
            for kind, directions in kinds.items():
                for d, value in directions.items():
                    force_rows.append(dict(candidate=r['id'], budget=budget, kind=kind, direction=d,
                        status=value['status'], capacity_n=value['capacity_n'],
                        limiting_joints=';'.join(value.get('limiting_joint_names', []))))
        for mode, directions in r['instantaneous'].items():
            for d, value in directions.items():
                instant_rows.append(dict(candidate=r['id'], mode=mode, direction=d, **{
                    k:value.get(k) for k in ['status', 'speed_m_s', 'min_task_singular_value', 'rank']}))
    write_csv(out/'force.csv', force_rows); write_csv(out/'instantaneous.csv', instant_rows)
    correlation = []; populations = []; pair_populations = []
    for h in config['motion']['horizons_s']:
        for mobile in [False, True]:
            for d in DIRECTIONS:
                rows = [r for r in reach if r['horizon_s']==h and r['mobile']==mobile and r['direction']==d]
                populations.append(dict(horizon_s=h, mobile=mobile, direction=d,
                    **describe([r['achieved_displacement_m'] for r in rows])))
                pc = [x for p in metrics for x in p['comparisons']
                      if x['horizon_s']==h and x['mobile']==mobile and x['direction']==d]
                pair_populations.append(dict(horizon_s=h, mobile=mobile, direction=d,
                    count=len(pc), practical_count=sum(x['practical_difference'] for x in pc),
                    **{'gap_'+k:v for k,v in describe([x['gap_m'] for x in pc]).items()}))
                if h==.3:
                    xs = [r['force']['nominal']['residual'][d]['capacity_n'] for r in accepted]
                    ys = [lookup[(r['id'],h,mobile,d)]['achieved_displacement_m'] for r in accepted]
                    valid = [(x,y) for x,y in zip(xs,ys) if x is not None and y is not None]
                    coefficient = float(spearmanr(*zip(*valid)).statistic) if len(valid)>2 else None
                    correlation.append(dict(mobile=mobile,direction=d,n=len(valid),spearman_rho=coefficient,
                        interpretation='descriptive only; no independent-sample p value, no hardware inference'))
    write_csv(out/'population.csv', populations); write_csv(out/'pair_population.csv', pair_populations)
    base_gains = [dict(candidate=r['candidate'],horizon_s=r['horizon_s'],direction=r['direction'],
        fixed_m=lookup[(r['candidate'],r['horizon_s'],False,r['direction'])]['achieved_displacement_m'],
        mobile_m=r['achieved_displacement_m'],
        gain_m=r['achieved_displacement_m']-lookup[(r['candidate'],r['horizon_s'],False,r['direction'])]['achieved_displacement_m'])
        for r in reach if r['mobile'] and r['achieved_displacement_m'] is not None
        and lookup[(r['candidate'],r['horizon_s'],False,r['direction'])]['achieved_displacement_m'] is not None]
    write_csv(out/'base_gains.csv', base_gains)
    force_status = Counter(x['status'] for r in accepted for kinds in r['force'].values() for ds in kinds.values() for x in ds.values())
    summary = dict(interpretation='Offline local path-family bounds, URDF geometry plus nominal/assumed constraints; not H1 hardware or human recovery validation',
        candidate_statuses=dict(Counter(r['status'] for r in force)),matched_pairs=len(metrics),
        total_pairs=len(read('all_force_pairs.json')),force_lp_statuses=dict(force_status),
        main=summarize_cases(reach),population=populations,pair_population=pair_populations,
        correlation=correlation,base_gain_m=describe([r['gain_m'] for r in base_gains]),
        pairs_with_practical_difference_fixed_03=sum(p['primary_practical_directions']>0 for p in metrics),
        primary_mean_gap_m=describe([p['primary_mean_six_direction_gap_m'] for p in metrics]),
        selection_rule='Top three disjoint pairs by mean absolute achieved-displacement gap across ALL SIX directions at H=0.3s, fixed base; descriptive outcome selection, not held-out validation',
        selected_ids=[[p['a'],p['b']] for p in selected],
        max_initial_position_error_m=max(e['position_m'] for r in accepted for e in r['target_errors']),
        max_initial_rotation_error_rad=max(e['rotation_rad'] for r in accepted for e in r['target_errors']),
        shared_torso_task_abs_peak_nm=max(abs(x) for r in accepted for x in r['shared_torso_torque_nm']),
        shared_torso_capacity_abs_peak_nm=max(abs(x) for r in accepted for kinds in r['force'].values()
            for ds in kinds.values() for x0 in ds.values() for i,x in enumerate(x0.get('compensating_torque_nm',[]))
            if i in shared_indices))
    if (out/'sensitivity.json').exists():
        sensitivity=read('sensitivity.json'); summary['sensitivity']=summarize_cases(sensitivity)
        variants=[]
        for name in sorted(set(r['experiment'] for r in sensitivity)):
            for mobile in [False,True]:
                rows=[r for r in sensitivity if r['experiment']==name and r['mobile']==mobile]
                if not rows:continue
                changes=[]
                for r in rows:
                    nominal=lookup[(r['candidate'],r['horizon_s'],mobile,r['direction'])]
                    v=r['achieved_displacement_m']; nv=nominal['achieved_displacement_m']
                    changes.append(None if v is None or nv is None else v-nv)
                variants.append(dict(experiment=name,mobile=mobile,**summarize_cases(rows),change_from_nominal_m=describe(changes)))
        summary['sensitivity_variants']=variants
        sl={(r['experiment'],r['candidate'],r['mobile'],r['direction']):r for r in sensitivity}
        comparisons=[]
        for p in selected:
            for tag in ['main']+sorted(set(r['experiment'] for r in sensitivity)):
                for mobile in [False,True]:
                    for d in DIRECTIONS:
                        if tag=='main':a=lookup[(p['a'],.3,mobile,d)];b=lookup[(p['b'],.3,mobile,d)]
                        else:
                            a=sl.get((tag,p['a'],mobile,d));b=sl.get((tag,p['b'],mobile,d))
                        if a is None or b is None:continue
                        va,vb=a['achieved_displacement_m'],b['achieved_displacement_m']
                        comparisons.append(dict(a=p['a'],b=p['b'],experiment=tag,mobile=mobile,direction=d,
                            a_m=va,b_m=vb,gap_m=None if va is None or vb is None else abs(va-vb),
                            winner=None if va is None or vb is None else p['a'] if va>vb else p['b'] if vb>va else 'tie',
                            a_status=a['status'],b_status=b['status']))
        write_csv(out/'selected_pair_sensitivity.csv',comparisons)
    dump(out/'summary.json', summary)
    inputs=['config.yaml','force_and_instantaneous.json','all_force_pairs.json','reachability.json']
    if (out/'sensitivity.json').exists():inputs.append('sensitivity.json')
    dump(out/'analysis_metadata.json',dict(
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(),
        input_hashes={name:sha256(out/name) for name in inputs},selection_rule=summary['selection_rule']))
    if plots:
        from .plots import generate
        generate(out)
    print(json.dumps({k:summary[k] for k in ['matched_pairs','selected_ids','pairs_with_practical_difference_fixed_03','primary_mean_gap_m']},indent=2))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();analyze(args.output,not args.no_plots)
