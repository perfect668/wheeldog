"""Export selective C2 review evidence; raw assets/frames/solver logs stay native."""
import argparse,gzip,hashlib,json,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--tar-path',type=Path,default=Path('/tmp/dynadress-c2-review.tar.gz'));a=p.parse_args()
    plan_path=ROOT/'dynadress/configs/c2_paired/plan.json';plan=json.loads(plan_path.read_text())
    metrics=json.loads((ROOT/'artifacts/c2/evaluation/metrics.json').read_text())
    if len(metrics['episodes'])!=36:raise ValueError('36 completed paired episodes required, including all failures')
    pinned=json.loads((ROOT/'docs/M21_SOURCE_MANIFEST.json').read_text())['files']
    model_sources={}
    for name in ('dynadress/models/structured_world_model.py','dynadress/models/impedance.py'):
        model_sources[name]=sha(ROOT/name)
        if model_sources[name]!=pinned[name]['sha256']:raise ValueError('M2.1 model source is no longer frozen')
    frozen=None;runtime_sources=None;provenance=[]
    for e in plan['episodes']:
        path=ROOT/'artifacts/c2/paired'/e['episode_id'];meta=json.loads((path/'metadata.json').read_text())
        if not meta['complete'] or meta['seed']!=plan['seed']:raise ValueError('Incomplete/mismatched paired seed')
        cfg=meta['config']
        if cfg['initial_offset_m']!=plan['initial_offset_m']:raise ValueError('Paired offset mismatch')
        if cfg['human']['stiffness']!=[e['K']]*4 or cfg['human']['damping']!=[e['D']]*4 or cfg['garment']['friction']!=e['mu']:
            raise ValueError('Manifest and actual physics configuration disagree')
        hashes=meta['c2_source_sha256']
        if frozen is None:frozen=hashes
        if hashes!=frozen:raise ValueError('Frozen C2 model/inference/calibration changed within the paired study')
        if runtime_sources is None:runtime_sources=meta['code_sha256']
        if meta['code_sha256']!=runtime_sources:raise ValueError('Native runtime source changed within the paired study')
        provenance.append(dict(episode_id=e['episode_id'],seed=meta['seed'],project_commit=meta['project_commit'],
            isaac_version=meta['isaac_version'],code_sha256=meta['code_sha256'],metadata_sha256=sha(path/'metadata.json'),
            result_sha256=sha(path/'result.json')))
    selected=[]
    for folder in ('evaluation','replay','replay_fixed_mu'):
        directory=ROOT/'artifacts/c2'/folder
        for path in directory.glob('*.json'):
            if path.name=='metrics.json' or not (path.name.startswith('e0') or path.name.endswith('_beliefs.json')):
                selected.append(path);continue
            target=Path(str(path)+'.gz');target.write_bytes(gzip.compress(path.read_bytes(),compresslevel=9,mtime=0));selected.append(target)
        for pattern in ('*.npz','*.png','*.pdf','*.csv','*_actions.jsonl'):
            selected.extend(directory.glob(pattern))
    for folder in ('calibration','probe_calibration'):
        selected.extend((ROOT/'artifacts/c2'/folder).glob('*.*'))
    for name in ('active_probe.mp4','active_probe.png','video_receipt.json'):
        selected.append(ROOT/'artifacts/c2/video'/name)
    video=json.loads((ROOT/'artifacts/c2/video/video_receipt.json').read_text())
    if video['seed']!=plan['seed']:raise ValueError('Video is from a different collection seed')
    logs=ROOT/'setup_logs/c2'
    for name in ('tests.log','review_contract_tests.log','smoke_receipt.log','evaluation.log','mu_diagnosis.log','plots.log','video.log'):
        if name=='review_contract_tests.log' and not (logs/name).exists():continue
        selected.append(logs/name)
    for name in ('paired.log','smoke_final.log'):
        path=logs/name;target=Path(str(path)+'.gz');target.write_bytes(gzip.compress(path.read_bytes(),9,mtime=0));selected.append(target)
    # Compact metadata records keep source provenance without raw high-dimensional assets.
    manifest=dict(base_commit='8cb2b3be62571629b0834aada8bf0255e0df169d',plan_sha256=sha(plan_path),
        paired_seed=plan['seed'],episodes=provenance,frozen_inference_sha256=frozen,frozen_model_source_sha256=model_sources,
        native_runtime_source_sha256=runtime_sources,
        exports={str(path.relative_to(ROOT)):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(set(selected))},
        note='Raw 20Hz solver/action logs, FEM assets and native RGB remain on server. Review sequences/actions/posteriors are complete for 36 episodes. Older overlapping-seed pilot data are excluded and retained separately.')
    manifest_path=ROOT/'docs/C2_SOURCE_MANIFEST.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');selected.append(manifest_path)
    with tarfile.open(a.tar_path,'w:gz') as tar:
        for path in sorted(set(selected)):tar.add(path,arcname=str(path.relative_to(ROOT)))
    receipt=dict(path=str(a.tar_path),bytes=a.tar_path.stat().st_size,sha256=sha(a.tar_path),files=len(set(selected)),
        raw_selected_bytes=sum(p.stat().st_size for p in set(selected)))
    (logs/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
