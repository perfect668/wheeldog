"""Finish an already-running acquisition, then run the real model pipeline.

This is a one-off task pipeline, not a scheduler. It refuses incomplete plans.
Startup experiments are kept separately; the two early real prototypes are
repeated with the frozen collector so the final 100 have full source hashes.
"""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--wait-for-pid',type=int);a=p.parse_args()
if a.wait_for_pid:
    proc=Path(f'/proc/{a.wait_for_pid}')
    while proc.exists():
        try:
            cmd=(proc/'cmdline').read_bytes()
            if b'm2_collect_episode.py' not in cmd:break
        except OSError:break
        time.sleep(20)
logs=ROOT/'setup_logs/m2';logs.mkdir(parents=True,exist_ok=True);records=[]
def command(name,args):
    start=time.time_ns();log=logs/f'{name}.log'
    with log.open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    records.append(dict(name=name,command=list(map(str,args)),returncode=r.returncode,start_wall_ns=start,end_wall_ns=time.time_ns(),log=str(log.relative_to(ROOT))))
    (logs/'final_pipeline.json').write_text(json.dumps(records,indent=2))
    if r.returncode:raise RuntimeError(f'{name} failed; inspect {log}')
    print('M2_PIPELINE',name,'PASS',flush=True)
plan=json.loads((ROOT/'dynadress/configs/m2_collection/plan.json').read_text())['episodes']
for item in plan:
    meta=ROOT/'artifacts/m2/episodes'/item['episode_id']/'metadata.json'
    if not meta.exists() or not json.loads(meta.read_text()).get('complete'):raise RuntimeError(f'Acquisition incomplete: {item["episode_id"]}')
for eid in ['e001_k0_d0_f0_r1','e007_k0_d0_f3_r1']:
    run=ROOT/'artifacts/m2/episodes'/eid;meta=json.loads((run/'metadata.json').read_text())
    if 'dynadress/interaction/wrist_load.py' not in meta['code_sha256']:
        archive=ROOT/'artifacts/m2/pilots'/f'initial_{eid}';archive.parent.mkdir(parents=True,exist_ok=True)
        run.rename(archive)
command('final_collect_two',['python3',str(ROOT/'scripts/m2_batch.py')])
command('wrist_replay',[str(ROOT/'scripts/run_native.sh'),'python',str(ROOT/'scripts/m2_wrist_replay.py'),str(ROOT/'artifacts/m2/episodes/e000_k0_d0_f0_r0')])
model_python=str(ROOT/'scripts/m2_model_python.sh')
for name,script in [('dataset','m2_dataset.py'),('validate','m2_validate_episodes.py')]:command('final_'+name,[model_python,str(ROOT/'scripts'/script)])
assert json.loads((ROOT/'artifacts/m2/dataset/validation.json').read_text())['complete_episodes']==100
command('final_train',[model_python,str(ROOT/'scripts/m2_train.py'),'--epochs','100'])
command('final_evaluate',[model_python,str(ROOT/'scripts/m2_evaluate.py')])
command('final_collection_analysis',[model_python,str(ROOT/'scripts/m2_analyze_collection.py')])
for item in plan:
    run=ROOT/'artifacts/m2/episodes'/item['episode_id']
    if (run/'frame_00000.jpg').exists():command('encode_'+item['episode_id'],['python3',str(ROOT/'scripts/m11_encode.py'),str(run)])
receipt=dict(complete=True,completed_wall_ns=time.time_ns(),episodes=100,
             model_sha256=hashlib.sha256((ROOT/'artifacts/m2/model/best.pt').read_bytes()).hexdigest(),commands=records)
(ROOT/'artifacts/m2/completion_receipt.json').write_text(json.dumps(receipt,indent=2))
print('M2_FINAL_PIPELINE_COMPLETE',flush=True)
