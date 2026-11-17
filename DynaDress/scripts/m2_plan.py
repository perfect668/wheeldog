"""Generate a frozen, independent K x D x friction factorial (100 episodes)."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'dynadress/configs/m2_collection';out.mkdir(parents=True,exist_ok=True)
base=json.loads((ROOT/'dynadress/configs/m2_pilot.json').read_text())
rng=np.random.default_rng(20261117);index=[]
for ki,k in enumerate([4.,8.,12.,16.,20.]):
    for di,d in enumerate([.5,2.]):
        for fi,mu in enumerate([.1,.2,.3,.5,.8]):
            combo=f'k{ki}_d{di}_f{fi}'
            for repeat in range(2):
                cfg=json.loads(json.dumps(base));eid=f'e{len(index):03d}_{combo}_r{repeat}'
                cfg['episode_id']=eid;cfg['output_group']='m2/episodes'
                cfg['parameter_group']=combo;cfg['seed']=20261117+len(index)
                cfg['human']['stiffness']=[k]*4;cfg['human']['damping']=[d]*4
                cfg['garment']['friction']=mu
                cfg['action_profile']='baseline' if repeat==0 else ['probe_a','probe_b','probe_c'][(ki+di+fi)%3]
                # Factorial properties are independent; alignment does not encode K/D/mu.
                cfg['initial_offset_m']=[float(rng.uniform(-.002,.002)),float(rng.uniform(-.008,.008)),float(rng.uniform(-.002,.002))]
                if len(index)%19==18:cfg['initial_offset_m'][1]=.020 # documented stress, keep failure
                cfg['rgb_hz']=2;cfg['record_rgb']=len(index) in [0,1,18,51]
                path=out/f'{eid}.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
                index.append(dict(episode_id=eid,parameter_group=combo,config=str(path.relative_to(ROOT)),K=k,D=d,friction=mu,profile=cfg['action_profile'],offset_m=cfg['initial_offset_m']))
(out/'plan.json').write_text(json.dumps(dict(seed=20261117,episodes=index),indent=2)+'\n')
print(f'Frozen plan: {len(index)} episodes / {len(set(x["parameter_group"] for x in index))} parameter groups')
