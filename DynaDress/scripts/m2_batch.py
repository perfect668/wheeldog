"""Resume the frozen plan using one Kit process and fresh scenes per episode.

Concurrent Isaac startup stalled in this server's native installation; the
supported collector serializes scene reconstruction. Offline model tools may
run concurrently because they do not instantiate Kit or modify USD stages.
"""
import argparse,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args()
cmd=[str(ROOT/'scripts/run_native.sh'),'python',str(ROOT/'scripts/m2_collect_episode.py'),
     '--batch-plan',str(ROOT/'dynadress/configs/m2_collection/plan.json')]
if a.limit:cmd.extend(['--limit',str(a.limit)])
raise SystemExit(subprocess.call(cmd))
