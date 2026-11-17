"""Audit the six pre-M2 episodes without inventing their absent load channel."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ids=['20261117T144712_trajectory02','20261117T145136_batch_K4_y-0.005','20261117T145459_batch_K4_y+0.005','20261117T145818_batch_K12_y+0.000','20261117T150136_batch_K12_y+0.005','20261117T150534_stress_K12_y+0.020']
results=[]
for eid in ids:
    run=ROOT/'artifacts/m11'/eid;meta=json.loads((run/'metadata.json').read_text());rows=[json.loads(r) for r in (run/'samples.jsonl').read_text().splitlines()]
    np.testing.assert_allclose(np.diff([r['simulation_time_s'] for r in rows]),np.diff([r['physics_step'] for r in rows])/240,atol=2e-6)
    assert meta['complete'] and all(not r['estimated_load']['available'] and r['estimated_load']['value'] is None for r in rows)
    results.append(dict(episode=eid,samples=len(rows),outcome=meta['outcome'],clock_valid=True,load_available=False,training_use='excluded from load supervision; never zero-imputed'))
out=ROOT/'artifacts/m2/legacy_audit.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
