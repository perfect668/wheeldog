from pathlib import Path
import copy
import json
import subprocess
import importlib.metadata
import numpy as np
import yaml
from scipy.interpolate import CubicHermiteSpline
from h1_kinematics.assets import sha256,dump
from h1_capability.experiment import setup,metadata_record
from h1_coupled.experiment import load_config as coupled_config,check_frozen
from h1_coupled.chart import SpatialChart


def load_config(path='configs/m2_posture.yaml'):
    c=yaml.safe_load(Path(path).read_text());c['_coupled']=coupled_config(c['coupled_config'])
    c['_coupled']['experiment']['dt_s']=c['control']['dt_s']
    return c


def variant(c,name):
    cc=copy.deepcopy(c['_coupled'])
    for key,value in c['evaluation']['variants'][name].items():
        if key in cc['contact']:cc['contact'][key]=value
        elif key in cc['_profile']:cc['_profile'][key]=value
        else:raise ValueError(key)
    cc['experiment']['validation_dt_s']=c['evaluation']['validation_dt_s']
    cc['experiment']['collision_stride']=c['evaluation']['collision_stride']
    return cc


def frozen():
    report=check_frozen();inventory=Path('results/m1d/artifact_sha256.json')
    hashes=json.loads(inventory.read_text())
    for p,h in hashes.items():
        if sha256(p)!=h:raise ValueError('M1-D artifact changed: '+p)
    report[str(inventory)]=dict(count=len(hashes),all_match=True,inventory_sha256=sha256(inventory))
    return report


def provenance(out,stage):
    dump(Path(out)/(stage+'_provenance.json'),dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tracked_status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip(),
        started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))


class Context:
    def __init__(self,c,out):
        self.c=c;self.out=Path(out);_,self.robot,self.checker=setup(c['_coupled']['_capability'])
        self.records={r['id']:r for r in json.loads(Path(c['_coupled']['_capability']['candidates']).read_text())}
        self.cache={}

    def chart(self,candidate,policy,cc):
        # No rebuilding FK/J; load the immutable prepared geometric curve.
        key=(candidate,policy,json.dumps(cc['contact'],sort_keys=True),json.dumps(cc['_profile'],sort_keys=True))
        if key in self.cache:return self.cache[key]
        options=copy.deepcopy(cc);options['chart']['inverse_policy']=policy
        chart=SpatialChart(self.robot,self.checker,self.records[candidate],options,False)
        p=self.out/'charts'/f'{candidate}_{policy}.npz'
        with np.load(p,allow_pickle=False) as d:chart.knots=d['knots_m'];chart.values=d['coordinates'];chart.slopes=d['slopes_per_m']
        chart.spline=CubicHermiteSpline(chart.knots,chart.values,chart.slopes)
        env=[];rejected=[]
        for radius in cc['chart']['domains_m']:
            try:env.append(chart.envelope(radius,cc['contact'],cc['robot']['torso_budget_nm']))
            except (ValueError,FloatingPointError,np.linalg.LinAlgError) as e:rejected.append(dict(radius_m=radius,reason=str(e)))
        self.cache[key]=(chart,env,rejected);return self.cache[key]

    def controller_chart(self,candidate,cc):
        entries=[];rejected=[]
        for policy in self.c['policies']:
            chart,env,bad=self.chart(candidate,policy,cc);rejected+=bad
            entries += [(chart,e,policy) for e in env]
        if not entries:raise ValueError('No initial-task admissible chart for '+candidate)
        chart,e,policy=max(entries,key=lambda x:x[1]['speed_m_s']*x[1]['acceleration_m_s2']*(x[1]['y_upper_m']-x[1]['y_lower_m']))
        return chart,e,policy,rejected


def prepare(c,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.yaml').exists():raise ValueError('prepare requires new output')
    (out/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False));(out/'charts').mkdir()
    m=metadata_record(c['_coupled']['_capability']);m['seed']=c['seed'];m['casadi']=importlib.metadata.version('casadi')
    m['scope']='M2-A fixed-base posture selection, nominal contact-only robot budgets and M1-D human contact model'
    m['gravity_model']='Robot gravity excluded; human effective gravity includes explicit upward contact load'
    m['frozen_integrity']=frozen();m['config_sha256']=sha256('configs/m2_posture.yaml');dump(out/'metadata.json',m)
    context=Context(c,out);inventory=[]
    for candidate in c['candidates']:
        r=context.records[candidate]
        if r['status']!='ACCEPTED_CONDITIONAL':raise ValueError('source configuration rejected')
        for policy in c['policies']:
            cc=copy.deepcopy(c['_coupled']);cc['chart']['inverse_policy']=policy
            chart=SpatialChart(context.robot,context.checker,r,cc,False).build()
            chart.save(out/'charts'/f'{candidate}_{policy}.npz',[])
            inventory.append(dict(candidate=candidate,policy=policy,rejected=chart.rejected,
                y_range_m=[float(chart.knots[0]),float(chart.knots[-1])]))
            print('prepared',candidate,policy,inventory[-1]['y_range_m'],flush=True)
    dump(out/'chart_inventory.json',inventory)
