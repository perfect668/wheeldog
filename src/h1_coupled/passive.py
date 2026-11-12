"""Stationary-handle physical baseline, same bilateral contact model."""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import numpy as np
import yaml
from h1_kinematics.assets import dump
from recoverability.disturbance import Disturbance
from .experiment import init_worker,get_chart
from .contact import CoupledModel
from .validation import validate


def run(output):
    out=Path(output);c=yaml.safe_load((out/'config.yaml').read_text());init_worker(c,out)
    rows=[];(out/'passive_paths').mkdir(exist_ok=True)
    for candidate in ['ik0024','ik0036']:
        _,chart,envelopes=get_chart(candidate,'fixed');e=envelopes[0]
        for stiffness in [150.,250.,400.]:
            cc=copy.deepcopy(c);cc['contact']['stiffness_per_hand_n_m']=stiffness
            model=CoupledModel(cc['_human'],cc['contact'],cc['experiment']['dt_s']);n=model.human.steps(1.6)
            for dv in [-.24,-.16,.16,.24]:
                states=np.zeros((n+1,4));states[0,1]=dv
                for k in range(n):states[k+1]=model.step(states[k],0)
                name=f'passive_{candidate}_{stiffness:g}_{dv:+.2f}'
                metric=validate(model,chart,e,states,np.zeros(n),Disturbance(delta_v_m_s=dv),1.6,cc,out/'passive_paths'/(name+'.npz'))
                rows.append(dict(case_id=name,candidate=candidate,stiffness_n_m=stiffness,impulse_m_s=dv,
                    valid=metric['valid'],recovered=metric['valid'] and metric['terminal_met'],metrics=metric))
    dump(out/'passive.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        information='No control and no handle motion; same contact constraints; failures are model/controller outcomes',cases=rows))
    print('passive',len(rows),'recovered',sum(r['recovered'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');run(p.parse_args().output)
