"""Bounded host/process/GPU sampler; RSS excludes shared-page deduplication."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,default=120);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
    for _ in range(a.seconds):
        processes=[]
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if '/kit/python/bin/python3' not in cmd or 'scripts/m1_' not in cmd:continue
                stat=(proc/'stat').read_text().split(') ',1)[1].split()
                rss=next(int(x.split()[1])*1024 for x in (proc/'status').read_text().splitlines() if x.startswith('VmRSS:'))
                processes.append(dict(pid=int(proc.name),rss_bytes=rss,utime_ticks=int(stat[11]),stime_ticks=int(stat[12]),command=cmd))
            except (OSError,StopIteration):continue
        gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
        row=dict(wall_time_ns=time.time_ns(),gpu_memory_MiB_and_utilization_percent=gpu,processes=processes)
        f.write(json.dumps(row)+'\n');f.flush();time.sleep(1)
