import json
import numpy as np
import yaml
from recoverability.experiment import run
from recoverability.plots import load_rows, boundary_rows


def test_raw_data_reproducible_and_self_describing(tmp_path):
    config=yaml.safe_load(open('configs/exp0.yaml'))
    config['actuators']={'A_fast':config['actuators']['A_fast']}
    config['sweep']={'delta_v_min_m_s':-.2,'delta_v_max_m_s':.2,'delta_v_step_m_s':.2,'horizons_s':[.4]}
    config['feedback']['horizons_s']=[.4]
    config['cases']=[]
    first,second=tmp_path/'one',tmp_path/'two'
    run(config,first,seed=17);run(config,second,seed=17)
    rows=load_rows(first)
    assert len(rows)==6
    assert all(r['seed']=='17' and r['model_json'] and r['actuator_json'] for r in rows)
    with np.load(first/'trajectories.npz',allow_pickle=False) as a,np.load(second/'trajectories.npz',allow_pickle=False) as b:
        for key in a.files:
            if key.endswith(('_state','_force','_command')):np.testing.assert_array_equal(a[key],b[key])
        assert json.loads(str(a['metadata_json']))['versions']['scipy']
        assert len(a['rows_json'])==6
    assert boundary_rows(rows)
