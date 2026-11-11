"""Read-only check of the pre-M1 frozen artifact inventory."""
import hashlib
import json
from pathlib import Path

def verify():
    entries=json.loads(Path('results/diagnostics/artifact_sha256.json').read_text())
    rows=[dict(path=p,expected=h,actual=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p,h in entries.items()]
    return dict(inventory='results/diagnostics/artifact_sha256.json',count=len(rows),
                all_match=all(r['expected']==r['actual'] for r in rows),files=rows)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args()
    result=verify();print(json.dumps(dict(count=result['count'],all_match=result['all_match'])))
    if a.output: Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_match']: raise SystemExit(1)
