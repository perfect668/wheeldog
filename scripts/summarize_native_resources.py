#!/usr/bin/env python3
"""Summarize the actual five-second native deployment resource samples."""
import datetime as dt
import json
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
blocks = re.split(r'(?m)^(\d{4}-\d\d-\d\dT[^\n]+)\n', text)
samples = []
for i in range(1, len(blocks)-1, 2):
    timestamp, body = blocks[i:i+2]
    lines = body.splitlines()
    if len(lines) < 3:
        continue
    gpu = re.match(r'(\d+) %, (\d+) MiB, (\d+) MiB, ([\d.]+) W', lines[0])
    cpu = re.search(r'^usage_usec (\d+)$', body, re.M)
    if not gpu or not lines[1].isdigit() or not cpu:
        continue
    rows = []
    for line in lines:
        p = re.match(r'\s*(\d+)\s+([\d.]+)\s+(\d+)\s+python3$', line)
        if p:
            rows.append(int(p[3]))
    samples.append(dict(timestamp=timestamp, gpu_util_percent=int(gpu[1]),
                        gpu_memory_mib=int(gpu[2]), gpu_total_mib=int(gpu[3]),
                        gpu_power_w=float(gpu[4]), cgroup_memory_bytes=int(lines[1]),
                        cpu_usage_usec=int(cpu[1]), max_python_rss_kib=max(rows, default=0)))
if not samples:
    raise SystemExit('No complete resource samples')
cores = []
for a,b in zip(samples,samples[1:]):
    elapsed = (dt.datetime.fromisoformat(b['timestamp'])-dt.datetime.fromisoformat(a['timestamp'])).total_seconds()
    if elapsed > 0:
        cores.append((b['cpu_usage_usec']-a['cpu_usage_usec'])/elapsed/1e6)
result = {
    'scope': 'whole instance cgroup and GPU, including desktop/Jupyter and startup/shader compilation; not isolated steady-state benchmark',
    'sample_count': len(samples), 'sampling_period_s': 5,
    'start': samples[0]['timestamp'], 'end': samples[-1]['timestamp'],
    'peak_cgroup_memory_gib': max(s['cgroup_memory_bytes'] for s in samples)/2**30,
    'peak_python_rss_gib': max(s['max_python_rss_kib'] for s in samples)/2**20,
    'peak_gpu_memory_mib': max(s['gpu_memory_mib'] for s in samples),
    'peak_gpu_util_percent': max(s['gpu_util_percent'] for s in samples),
    'peak_gpu_power_w': max(s['gpu_power_w'] for s in samples),
    'peak_cpu_cores_by_interval': max(cores, default=None),
    'average_cpu_cores_by_interval': sum(cores)/len(cores) if cores else None,
}
print(json.dumps(result, indent=2))
