"""Directed native PhysX reruns around the retained e088 regression condition.

Run with run_native.sh (Kit is started by the preserved collector):
  ./scripts/run_native.sh python /root/gpufree-data/dynadress/scripts/m21_highfreq.py
Use --prepare-only with ordinary Python to inspect the three concrete configs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config',type=Path,default=ROOT/'dynadress/configs/m2_collection/e088_k4_d0_f4_r0.json')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--variant-offset-y-m',type=float,default=.002)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--plan-name',default='e088_directed')
    a,kit_args=p.parse_known_args()
    source=a.base_config.read_bytes();base=json.loads(source)
    directory=ROOT/'dynadress/configs/m21_highfreq'/a.plan_name
    directory.mkdir(parents=True,exist_ok=True)
    runs=[]
    for i in range(a.repeats+1):
        cfg=copy.deepcopy(base)
        variant=i==a.repeats
        eid=f'e088_hf_{"offset" if variant else "repeat"}_{i:02d}'
        cfg['episode_id']=eid
        cfg['output_group']='m21/highfreq/'+a.plan_name
        cfg['record_rgb']=False
        cfg['cloth_hz']=0
        if variant:cfg['initial_offset_m'][1]+=a.variant_offset_y_m
        cfg['m21_diagnostic']=dict(base_episode=base['episode_id'],
            base_config_sha256=hashlib.sha256(source).hexdigest(),
            status='new diagnostic rerun; excluded from original training/validation/test splits',
            variation='initial_offset_y_m' if variant else 'none; matching original physical parameters and controller',
            variation_m=a.variant_offset_y_m if variant else 0.)
        path=directory/(eid+'.json')
        text=json.dumps(cfg,indent=2)+'\n'
        if path.exists() and path.read_text()!=text:raise RuntimeError(f'Refusing to overwrite changed config: {path}')
        path.write_text(text)
        runs.append(dict(episode_id=eid,config=str(path.relative_to(ROOT))))
    plan=directory/'plan.json'
    plan.write_text(json.dumps(dict(source_episode=base['episode_id'],episodes=runs),indent=2)+'\n')
    print('M21_HIGHFREQ_PLAN',plan,flush=True)
    if a.prepare_only:return
    sys.argv=[str(ROOT/'scripts/m2_collect_episode.py'),'--batch-plan',str(plan),'--high-frequency','--no-rgb',*kit_args]
    runpy.run_path(str(ROOT/'scripts/m2_collect_episode.py'),run_name='__main__')


if __name__=='__main__':main()
