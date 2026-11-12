"""Independent saved-data audit and denser replay; never writes old milestones."""
from pathlib import Path
import argparse
import copy
import json
import subprocess
import numpy as np
import yaml
from h1_kinematics.assets import dump
from h1_coupled.experiment import check_frozen,init_worker,get_chart,variant_config
from h1_coupled.contact import CoupledModel
from h1_coupled.validation import validate
from h1_coupled.direct import replay
from recoverability.disturbance import Disturbance


def run(output):
    out=Path(output);c=yaml.safe_load((out/'config.yaml').read_text());init_worker(c,out)
    report=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen=check_frozen(),saved_cases=0,accepted=0,recorded_force_error_n=0.,recorded_dynamics_error=0.)
    for path in sorted((out/'cases').glob('*.json')):
        r=json.loads(path.read_text());report['saved_cases']+=1
        if not r['status'].startswith('RECOVERED'):continue
        cc=variant_config(c,r['variant']);m=CoupledModel(cc['_human'],cc['contact'],cc['experiment']['dt_s'])
        with np.load(r['path_file'],allow_pickle=False) as d:
            actual= m.k*(d['handle_y_m']-d['state'][:,:1])+m.c*(d['handle_v_m_s']-d['state'][:,1:2])
            err=float(np.max(abs(actual-d['per_hand_lateral_force_n'])))
            report['recorded_force_error_n']=max(report['recorded_force_error_n'],err)
            assert err<1e-10
            for k,a in enumerate(d['acceleration_m_s2']):
                predicted=m.step(d['node_state'][k],a,d['disturbance_force_n'][k]);predicted[1]+=d['jumps_m_s'][k+1]
                err=float(max(abs(predicted-d['node_state'][k+1])))
                report['recorded_dynamics_error']=max(report['recorded_dynamics_error'],err);assert err<1e-7
            assert np.isfinite(d['q_rad']).all() and r['metrics']['valid'] and r['metrics']['terminal_met']
        report['accepted']+=1
    dense=[];main=json.loads((out/'main.json').read_text())
    selected=[r for r in main if r['candidate'] in ['ik0024','ik0036'] and r['horizon_s']==1.6
        and r['disturbance']['delta_v_m_s']==.16 and r['status'].startswith('RECOVERED')]
    for r in selected:
        cc,chart,_=get_chart(r['candidate'],r['mode']);cc=copy.deepcopy(cc);cc['experiment']['validation_dt_s']=.0025
        m=CoupledModel(cc['_human'],cc['contact'],cc['experiment']['dt_s'])
        with np.load(r['path_file'],allow_pickle=False) as d:
            e=json.loads(str(d['envelope_json']));v=validate(m,chart,e,d['node_state'],d['acceleration_m_s2'],Disturbance(**r['disturbance']),r['horizon_s'],cc)
        assert v['valid'] and v['terminal_met'];dense.append(dict(case_id=r['case_id'],validation=v))
    report['dense_2_5ms_replays']=dense
    direct=[]
    for r in json.loads((out/'direct.json').read_text()):
        if not r['path_file']:continue
        cc,chart,_=get_chart(r['candidate'],'fixed');cc=copy.deepcopy(cc);cc['experiment']['validation_dt_s']=.0025
        m=CoupledModel(cc['_human'],cc['contact'],cc['experiment']['dt_s'])
        with np.load(r['path_file'],allow_pickle=False) as d:
            v=replay(chart,m,d['node_states'],d['acceleration'],r['achieved_impulse_m_s'],r['sign'],cc)
        assert v['valid'];direct.append(dict(case_id=r['case_id'],validation=v))
    report['direct_dense_replays']=direct;dump(out/'validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');run(p.parse_args().output)
