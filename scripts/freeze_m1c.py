"""Audit provenance and preserve M1-C artifact hashes after a completed run."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from h1_capability.experiment import frozen_inventory
from h1_kinematics.assets import dump,sha256


def run(output,verify=False):
    root=Path(output);out=root/'run';inventory=root/'artifact_sha256.json'
    if verify:
        hashes=json.loads(inventory.read_text())
        for name,value in hashes.items():
            assert sha256(name)==value,f'Changed M1-C artifact: {name}'
        print('M1-C verified',len(hashes),'artifacts;',frozen_inventory())
        return
    read=lambda name:json.loads((out/name).read_text())
    main=read('reachability.json');sens=read('sensitivity.json');summary=read('summary.json')
    assert len(main)==1512 and len(sens)==720
    assert len({r['case_id'] for r in main+sens})==2232
    assert len(list((out/'cases').glob('*.json')))==2232
    assert len(list((out/'paths').glob('*.npz')))==2232
    assert read('replay_summary.json')['failures']==[]
    fixed={(r['candidate'],r['horizon_s'],r['direction']):r for r in main if not r['mobile']}
    for r in main:
        assert r['status']=='VERIFIED_PATH_FOUND'
        if r['mobile']:
            assert r['achieved_displacement_m']>=fixed[(r['candidate'],r['horizon_s'],r['direction'])]['achieved_displacement_m']-1e-12
    initial=read('metadata.json');kernel=initial['source_commit'];kernels={}
    for module in ['model','force','instantaneous','reachability','experiment','pairing']:
        path=f'src/h1_capability/{module}.py'
        original=subprocess.check_output(['git','show',f'{kernel}:{path}'])
        digest=hashlib.sha256(original).hexdigest()
        assert sha256(path)==digest,f'Calculation module changed after formal main experiment: {path}'
        kernels[path]=digest
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dump(out/'stage_provenance.json',dict(main_kernel_commit=kernel,
        sensitivity_commit=read('sensitivity_metadata.json')['source_commit'],
        analysis_commit=read('analysis_metadata.json')['source_commit'],
        replay_commit=read('replay_summary.json')['source_commit'],
        packaging_commit=source,kernel_files_sha256=kernels,
        calculation_kernel_unchanged_since_main=True,frozen_integrity=frozen_inventory(),
        case_count=len(main+sens),matched_pair_count=summary['matched_pairs'],
        interpretation='Later commits add analysis/tests/reports only; main calculation modules hash-identical'))
    hashes={str(p):sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=inventory}
    dump(inventory,hashes);print('Frozen',len(hashes),'M1-C artifacts')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--verify',action='store_true')
    a=p.parse_args();run(a.output,a.verify)
