"""Evaluate the SAME chart solver at each independent trajectory's exact impulse."""
from pathlib import Path
import argparse
import json
import subprocess
import yaml
from h1_coupled.experiment import init_worker,compute,cases_for
from h1_kinematics.assets import dump


def run(output):
    out=Path(output);c=yaml.safe_load((out/'config.yaml').read_text());init_worker(c,out);rows=[]
    direct_rows=json.loads((out/'direct.json').read_text())
    if (out/'direct_refined.json').exists():direct_rows+=json.loads((out/'direct_refined.json').read_text())
    for direct in direct_rows:
        if direct['achieved_impulse_m_s'] is None:continue
        dv=direct['sign']*direct['achieved_impulse_m_s']
        case=cases_for(c,out,candidates=[direct['candidate']],horizons=[direct['horizon_s']],impulses=[dv],modes=['fixed'],methods=['offline'])[0]
        case['case_id']='probe_'+direct['case_id'];case['path_file']=str(out/'paths'/(case['case_id']+'.npz'))
        r=compute(case);dump(out/'cases'/(r['case_id']+'.json'),r)
        rows.append(dict(direct_case=direct['case_id'],same_impulse_m_s=dv,chart_case=r['case_id'],chart_status=r['status'],
            direct_status=direct['status']))
    dump(out/'direct_chart_probes.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),cases=rows))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');run(p.parse_args().output)
