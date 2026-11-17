"""Small task dataset, gated on a genuinely successful nominal episode."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--nominal',type=Path,required=True);a=p.parse_args()
nom=json.loads((a.nominal/'result.json').read_text())
if nom.get('success') is not True or nom.get('termination_reason')!='geometric_completion':raise SystemExit('First provide a verified nominal sleeve episode; batch not started.')
logs=ROOT/'setup_logs/m11';logs.mkdir(parents=True,exist_ok=True)
index=[];stamp=time.strftime('%Y%m%dT%H%M%S')
for k,offset in [(4,-.005),(4,.005),(12,0.),(12,.005)]:
    tag=f'batch_K{k}_y{offset:+.3f}';before=set((ROOT/'artifacts/m11').iterdir());log=logs/f'{stamp}_{tag}.log'
    with log.open('w') as f:r=subprocess.run([str(ROOT/'scripts/run_native.sh'),'python',str(ROOT/'scripts/m1_coupled_pull.py'),'--stiffness',str(k),'--offset-y',str(offset),'--tag',tag],stdout=f,stderr=subprocess.STDOUT)
    runs=sorted(set((ROOT/'artifacts/m11').iterdir())-before)
    for run in runs:
        result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {'success':False,'termination_reason':'process_error; see ERROR.txt/log'}
        subprocess.run(['python3',str(ROOT/'scripts/m11_encode.py'),str(run)],check=False)
        index.append({'episode':str(run),'stiffness':k,'offset_y_m':offset,'process_returncode':r.returncode,'result':result,'log':str(log)})
    target=ROOT/'artifacts/m11'/f'batch_{stamp}.json';target.write_text(json.dumps(index,indent=2))
    print(json.dumps(index[-1]) if index else 'No output directory',flush=True)
