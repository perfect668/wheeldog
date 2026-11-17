"""Bounded acquisition resource/progress log. Stops after 100 finished episodes."""
import argparse,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--hours',type=float,default=5);a=p.parse_args()
out=ROOT/'setup_logs/m2/resources.jsonl';out.parent.mkdir(parents=True,exist_ok=True)
with out.open('a') as f:
    until=time.monotonic()+a.hours*3600
    while time.monotonic()<until:
        processes=[]
        for d in Path('/proc').glob('[0-9]*'):
            try:
                cmd=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if '/kit/python/bin/python3' not in cmd or '/scripts/m2_' not in cmd:continue
                stat=(d/'stat').read_text().split(') ',1)[1].split()
                rss=next(int(x.split()[1])*1024 for x in (d/'status').read_text().splitlines() if x.startswith('VmRSS:'))
                processes.append(dict(pid=int(d.name),rss_bytes=rss,cpu_ticks=int(stat[11])+int(stat[12]),command=cmd))
            except (OSError,StopIteration):continue
        results=[]
        for path in (ROOT/'artifacts/m2/episodes').glob('*/metadata.json'):
            try:
                meta=json.loads(path.read_text())
                if meta.get('complete'):results.append(meta)
            except (OSError,json.JSONDecodeError):pass
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True).strip()
        row=dict(wall_time_ns=time.time_ns(),gpu_MiB_percent_W=gpu,processes=processes,complete=len(results),successes=sum(r['outcome']=='success' for r in results))
        f.write(json.dumps(row)+'\n');f.flush()
        if len(results)>=100:break
        time.sleep(10)
