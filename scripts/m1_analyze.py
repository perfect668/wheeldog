"""Analyze only actual completed PhysX runs; never synthesize missing runs."""
import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',default='artifacts/m1');a=p.parse_args();root=Path(a.root)
selected={}
for path in sorted(root.glob('**/metadata.json')):
    meta=json.loads(path.read_text());result_path=path.with_name('result.json')
    if not meta.get('complete') or not result_path.exists():continue
    result=json.loads(result_path.read_text())
    if not result.get('clock_contract'):continue
    k=meta['stiffness'][0]
    condition='contact' if meta['cloth_present'] and meta['human_collision_enabled'] else ('collision_disabled' if meta['cloth_present'] else ('pulse' if meta.get('pulse_enabled',True) else 'free'))
    if condition=='pulse' and not result.get('passed'):continue
    selected[(condition,k)]=(path.parent,meta,result,np.load(path.with_name('response.npz'))['rows'])
out=root/'analysis';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(2,1,figsize=(9,7),constrained_layout=True)
for (condition,k),(directory,meta,result,data) in selected.items():
    error=data[:,3]-meta['q_ref'][1]
    ax=axes[0 if condition=='pulse' else 1]
    ax.plot(data[:,1],error,label=f'{condition}, K={k:g} Nm/rad',linewidth=1.5)
for ax in axes:
    ax.set_xlabel('Simulation time [s]');ax.set_ylabel('Shoulder Y displacement [rad]');ax.grid(alpha=.3);ax.legend(fontsize=8)
axes[0].set_title('Same 1 Nm joint-torque pulse (1.0–1.5 s), fixed reference')
axes[1].set_title('Same FEM shirt drop; only human collision / stiffness varied')
fig.savefig(out/'responses.png',dpi=170);fig.savefig(out/'responses.pdf')
manifest={'runs':{},'coupling_verdict':'incomplete','scope':'isolated cloth drop, not H1 pulling or sleeve insertion'}
for key,(directory,meta,result,data) in selected.items():manifest['runs'][str(key)]={'directory':str(directory),'max_deviation_rad':result['max_deviation_rad'],'final_error_rad':result['final_error_rad']}
if ('contact',4.) in selected and ('collision_disabled',4.) in selected:
    contact=selected[('contact',4.)][3];free=selected[('collision_disabled',4.)][3]
    if contact.shape!=free.shape or not np.allclose(contact[:,1],free[:,1],atol=1e-6):raise ValueError('Unmatched sampled simulation clock')
    delta=np.linalg.norm(contact[:,2:6]-free[:,2:6],axis=1)
    max_free=np.max(np.abs(free[:,2:6]-selected[('collision_disabled',4.)][1]['q_ref']))
    manifest.update(max_matched_contact_delta_rad=float(delta.max()),collision_disabled_max_error_rad=float(max_free),coupling_verdict='response_detected' if delta.max()>.001 and max_free<.0001 else 'not_demonstrated')
(out/'analysis.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
