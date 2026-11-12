"""Freeze/verify M1-D results by SHA256, including development failures and logs."""
from pathlib import Path
import argparse
import json
from h1_kinematics.assets import sha256,dump
from h1_coupled.experiment import check_frozen


def run(root,verify):
    root=Path(root);inventory=root/'artifact_sha256.json';check_frozen()
    if verify:
        hashes=json.loads(inventory.read_text())
        for path,digest in hashes.items():
            if sha256(path)!=digest:raise ValueError('changed artifact: '+path)
        current={str(p) for p in root.rglob('*') if p.is_file() and p!=inventory}
        if current!=set(hashes):raise ValueError('inventory paths differ from disk')
        print('M1-D verified',len(hashes),'artifacts; all previous milestone inventories also verified')
    else:
        if inventory.exists():raise ValueError('inventory already exists; use --verify')
        dump(inventory,{str(p):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=inventory})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',nargs='?',default='results/m1d');p.add_argument('--verify',action='store_true')
    a=p.parse_args();run(a.root,a.verify)
