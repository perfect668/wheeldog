"""Frozen paired C2 native experiments; no truth-dependent probe templates."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    label='smoke' if a.smoke else 'paired';out=ROOT/'dynadress/configs'/('c2_'+label)
    out.mkdir(parents=True,exist_ok=True)
    base=json.loads((ROOT/'dynadress/configs/m2_pilot.json').read_text())
    rows=[]
    combos=[(12.,.5,.1),(12.,.5,.8)] if a.smoke else [(k,d,mu) for k in (4.,12.,20.) for d in (.5,2.) for mu in (.1,.8)]
    for i,(k,d,mu) in enumerate(combos):
        for mode in ('passive','fixed','active'):
            cfg=json.loads(json.dumps(base));eid=f'c2_{label}_p{i:02d}_{mode}'
            cfg.update(episode_id=eid,output_group='c2/'+label,seed=20261119,
                initial_offset_m=[0.,.003,0.],action_profile='baseline',c2_mode=mode,
                max_seconds=25.,record_rgb=(mode=='active' and i==0),rgb_hz=8,cloth_hz=0)
            cfg['human']['stiffness']=[k]*4;cfg['human']['damping']=[d]*4;cfg['garment']['friction']=mu
            path=out/(eid+'.json');path.write_text(json.dumps(cfg,indent=2)+'\n')
            rows.append(dict(episode_id=eid,pair_id=f'p{i:02d}',mode=mode,config=str(path.relative_to(ROOT)),
                K=k,D=d,mu=mu,seed=cfg['seed'],initial_offset_m=cfg['initial_offset_m']))
    plan=dict(experiment='C2',seed=20261119,initial_offset_m=[0.,.003,0.],duration_s=25.,
        source_commit='8cb2b3be62571629b0834aada8bf0255e0df169d',episodes=rows,
        comparison='same observed contact+stable tracking trigger, 1.0s opportunity, original servo/termination')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(out/'plan.json',len(rows))
if __name__=='__main__':main()
