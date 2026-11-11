"""CPU-only deterministic experiments, complete raw data and provenance."""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import yaml
from .model import Model
from .actuator import Actuator
from .disturbance import Disturbance
from .solver import solve_oracle
from .feedback import simulate_feedback


def provenance(config, seed):
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            return 'unavailable'
    paths = sorted([p for directory in ('src','configs','scripts','tests') for p in (root/directory).rglob('*')
                    if p.is_file() and p.suffix in ('.py','.yaml')]+[root/'pyproject.toml'])
    files = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    versions = {}
    for package in ('numpy','scipy','matplotlib','PyYAML','pytest','wheeldog-recoverability'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    return {'created_utc':datetime.now(timezone.utc).isoformat(), 'config':config,'seed':seed,
            'deterministic':True,'git_commit':git('rev-parse','HEAD'),
            'git_status':git('status','--porcelain'),
            'implementation_status':git('status','--porcelain','--','src','configs','scripts','tests','pyproject.toml'),
            'source_sha256':files,'python':sys.version,'python_executable':sys.executable,
            'platform':platform.platform(),'cpu':platform.processor(),
            'versions':versions,'solver':'scipy.optimize.linprog / HiGHS IPM, explicit variable/row scaling',
            'interpretation':'toy simulation; known-disturbance offline oracle and separate causal fixed PD'}


class Dataset:
    def __init__(self, metadata):
        self.metadata, self.rows, self.arrays = metadata, [], {}

    def add(self, model, actuator, name, disturbance, horizon, method, result, family='sweep'):
        i = len(self.rows)
        row = {'case_id':f'c{i:05d}','family':family,'method':method,'actuator':name,
               'horizon_s':horizon,'dt_s':model.dt_s,**asdict(disturbance),
               'initial_x_m':0.0,'initial_v_m_s':0.0,'initial_force_N':0.0,
               'static_current_margin_N':actuator.force_max_N,
               'status':result.status,'recovery_criterion_met':result.metrics.get('valid_path_dynamics',False) and result.metrics.get('terminal_met',False),
               'solver_status':result.solver_status,'reason':result.reason,'elapsed_s':result.elapsed_s,
               'solver_time_s':result.solver_time_s,'seed':self.metadata['seed'],
               'git_commit':self.metadata['git_commit'],'model_json':json.dumps(asdict(model),sort_keys=True),
               'actuator_json':json.dumps(asdict(actuator),sort_keys=True),
               'config_json':json.dumps(self.metadata['config'],sort_keys=True),
               **result.metrics}
        self.rows.append(row)
        prefix = row['case_id']
        self.arrays[prefix+'_state'] = result.state
        self.arrays[prefix+'_force'] = result.force
        self.arrays[prefix+'_command'] = result.command
        jumps,d = disturbance.arrays(model,model.steps(horizon))
        self.arrays[prefix+'_jumps_m_s'],self.arrays[prefix+'_disturbance_N'] = jumps,d

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True,exist_ok=True)
        fields = list(dict.fromkeys(k for row in self.rows for k in row))
        with (output/'metrics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(self.rows)
        (output/'metadata.json').write_text(json.dumps(self.metadata,indent=2,ensure_ascii=False)+'\n')
        (output/'config.yaml').write_text(yaml.safe_dump(self.metadata['config'],sort_keys=False))
        # Standalone NPZ includes every per-case parameter, status, seed, and global versions/source hashes.
        np.savez_compressed(output/'trajectories.npz',**self.arrays,
                            rows_json=np.array([json.dumps(r,sort_keys=True) for r in self.rows]),
                            metadata_json=np.array(json.dumps(self.metadata,sort_keys=True)))
        counts = {}
        for row in self.rows:
            key = row['method']+':'+row['status']
            counts[key] = counts.get(key,0)+1
        elapsed=np.array([r['elapsed_s'] for r in self.rows])
        summary={'cases':len(self.rows),'counts':counts,
                 'case_elapsed_s_median':float(np.median(elapsed)),
                 'case_elapsed_s_p95':float(np.quantile(elapsed,.95)),
                 'case_elapsed_s_max':float(np.max(elapsed))}
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        return summary


def run(config, output, seed=0, mode='sweep'):
    metadata=provenance(config,seed)
    dataset=Dataset(metadata)
    model=Model(**config['model'])
    actuators={k:Actuator(**v) for k,v in config['actuators'].items()}
    budget=config.get('solver',{}).get('time_limit_s',5.0)
    started=time.perf_counter()
    if mode == 'sweep':
        sweep=config['sweep']
        lo,hi,step=[sweep[k] for k in ('delta_v_min_m_s','delta_v_max_m_s','delta_v_step_m_s')]
        if step <= 0 or hi < lo or not np.isclose((hi-lo)/step,round((hi-lo)/step)):
            raise ValueError('invalid disturbance grid')
        values=np.round(np.linspace(lo,hi,round((hi-lo)/step)+1),10)
        for name,actuator in actuators.items():
            for horizon in sweep['horizons_s']:
                for value in values:
                    disturbance=Disturbance(delta_v_m_s=float(value))
                    result=solve_oracle(model,actuator,disturbance,horizon,time_limit_s=budget)
                    dataset.add(model,actuator,name,disturbance,horizon,'oracle',result)
                    if horizon in config.get('feedback',{}).get('horizons_s',[]):
                        result=simulate_feedback(model,actuator,disturbance,horizon,
                            natural_frequency_s_inv=config['feedback']['natural_frequency_s_inv'])
                        dataset.add(model,actuator,name,disturbance,horizon,'causal_pd',result)
                print(f'{name} H={horizon:g}s complete',flush=True)
    for case in config.get('cases',[]):
        for name,actuator in actuators.items():
            disturbance=Disturbance(**case['disturbance'])
            horizon=case['horizon_s']
            for method in ('oracle','causal_pd'):
                result=(solve_oracle(model,actuator,disturbance,horizon,time_limit_s=budget) if method=='oracle'
                        else simulate_feedback(model,actuator,disturbance,horizon,
                          natural_frequency_s_inv=config.get('feedback',{}).get('natural_frequency_s_inv',4.0)))
                dataset.add(model,actuator,name,disturbance,horizon,method,result,family=case['name'])
    if not dataset.rows:
        raise ValueError('configuration produced no cases')
    metadata['run_elapsed_s']=time.perf_counter()-started
    summary=dataset.save(output)
    print(json.dumps(summary,indent=2),flush=True)
    return summary
