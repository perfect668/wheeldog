"""python -m h1_selection.experiment --stage ... --output results/m2a"""
from pathlib import Path
import argparse
import json
import yaml
from h1_kinematics.assets import dump,sha256
from .common import load_config,prepare,provenance,frozen


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/m2_posture.yaml')
    p.add_argument('--output',default='results/m2a');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--stage',required=True,choices=['prepare','score','select','test','analysis','verify','freeze'])
    a=p.parse_args();c=load_config(a.config);out=Path(a.output)
    if a.stage=='prepare':prepare(c,out);provenance(out,a.stage);return
    if yaml.safe_load((out/'config.yaml').read_text())!=c:raise ValueError('configuration differs from prepared experiment')
    if a.stage=='verify':
        result=dict(old=frozen())
        if (out/'artifact_sha256.json').exists():
            hashes=json.loads((out/'artifact_sha256.json').read_text())
            for path,h in hashes.items():
                if sha256(path)!=h:raise ValueError('M2 artifact changed: '+path)
            result['M2_artifacts_verified']=len(hashes)
        print(json.dumps(result,indent=2));return
    if a.stage=='freeze':
        dump(out/'artifact_sha256.json',{str(p):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_sha256.json'})
        return
    if a.stage=='test' and not (out/'selection.json').exists():raise ValueError('freeze selection before test')
    provenance(out,a.stage)
    if a.stage=='score':
        from .scoring import run
        run(c,out,a.workers)
    elif a.stage=='select':
        if (out/'test_selection_freeze.json').exists():raise ValueError('selection frozen for holdout')
        from .scoring import select
        select(c,out)
    elif a.stage=='test':
        from .evaluation import run
        run(c,out,a.workers)
    elif a.stage=='analysis':
        from .analysis import run
        run(c,out)


if __name__=='__main__':main()
