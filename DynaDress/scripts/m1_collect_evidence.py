"""Export existing runs selected from real logs, with lossless data and validated video."""
from pathlib import Path
import argparse,json,subprocess,tarfile
p=argparse.ArgumentParser();p.add_argument('logs',nargs='+');p.add_argument('--archive',default='/tmp/m1-final-evidence.tgz');a=p.parse_args()
root=Path(__file__).resolve().parents[1];runs=[]
for log in a.logs:
    for line in Path(log).read_text().splitlines():
        for marker in ['M1-HUMAN-RESULT ','M1-H1-MOTION-RESULT ','M1-COUPLED-PULL-RESULT ']:
            if marker in line:
                run=Path(line.split(marker,1)[1].split(' ',1)[0])
                if run.is_dir() and run not in runs:runs.append(run)
manifest=[]
with tarfile.open(a.archive,'w:gz') as tar:
    for run in runs:
        frames=sorted(run.glob('frame_*.png'));fps=20 if 'h1_motion' in str(run) else 10
        if frames:subprocess.run(['python3',str(root/'scripts/m1_encode.py'),str(run),'--fps',str(fps)],check=True)
        files=[f for f in run.iterdir() if f.is_file() and f.suffix!='.png' and not f.name.startswith('cloth_')]
        if frames:files.extend([frames[0],frames[len(frames)//2],frames[-1]])
        for f in sorted(set(files)):tar.add(f,arcname=str(f.relative_to(root)))
        manifest.append(str(run.relative_to(root)))
    for f in (root/'setup_logs/m1').glob('*.log'):tar.add(f,arcname=str(f.relative_to(root)))
(root/'setup_logs/m1/evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'archive':a.archive,'runs':manifest},indent=2))
