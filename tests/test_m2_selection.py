import copy
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from h1_selection.common import load_config
from h1_selection.control import PredictiveController,make_controller
from h1_selection.evaluation import ScheduledDisturbance,recovery_metrics,cases,actual_step,simulate
from h1_coupled.contact import CoupledModel


@pytest.fixture(scope='module')
def config_model():
    c=load_config();cc=c['_coupled'];return c,CoupledModel(cc['_human'],cc['contact'],.04)


def envelope(m):
    return dict(y_lower_m=-.12,y_upper_m=.12,speed_m_s=.3,acceleration_m_s2=1.,
        per_hand_force_lower_n=-m.cap_y,per_hand_force_upper_n=m.cap_y)


@pytest.mark.parametrize('controller',['fixed','mpc'])
def test_nonanticipative_and_exact_delay(config_model,controller):
    c,m=config_model;a=make_controller(controller,m,envelope(m),.08,c['control']);b=make_controller(controller,m,envelope(m),.08,c['control'])
    assert list(inspect.signature(a.command).parameters)==['observed']
    sa=np.zeros(4);sb=sa.copy();sent=[];delivered=[]
    for i in range(12):
        aa,la=a.command(sa);ab,lb=b.command(sb)
        if i<8:assert aa==pytest.approx(ab,abs=1e-10)
        sent.append(la['sent']);delivered.append(aa)
        sa=m.step(sa,aa);sb=m.step(sb,ab)
        if i==7:sb[1]+=.1
    assert delivered[:2]==[0.,0.]
    assert np.allclose(delivered[2:],sent[:-2],atol=1e-10)
    assert max(abs(sa))<1e-10 and np.max(abs(sb))>0


def test_mpc_satisfies_qp_and_recovers(config_model):
    c,m=config_model;p=PredictiveController(m,envelope(m),.04,c['control']);s=np.array([0,.08,0,0.])
    for i in range(80):
        a,l=p.command(s)
        assert a is not None and not l['fallback']
        assert l['qp_max_normalized_violation']<=c['control']['acceptance_abs_residual']
        assert abs(a)<=1+1e-7
        s=m.step(s,a)
        assert abs(s[2])<=.12+1e-7 and abs(s[3])<=.3+1e-7
    assert abs(s[0])<.01 and abs(s[1])<.03


def test_mpc_failure_uses_identical_pending_queue(config_model):
    c,m=config_model;p=PredictiveController(m,envelope(m),.08,c['control']);p.queue=[.1,.2]
    def broken(**kwargs):raise RuntimeError('injected solver failure')
    p.solver=broken
    delivered,log=p.command(np.zeros(4))
    assert log['fallback'] and log['qp_status']=='EXCEPTION' and delivered==.1
    assert p.queue[0]==.2 and log['governor']['delivered']==.1


def test_nonfinite_solver_output_is_rejected_and_log_is_serializable(config_model):
    c,m=config_model;p=PredictiveController(m,envelope(m),.04,c['control'])
    class NonfiniteSolver:
        def __call__(self,**kwargs):return {'x':np.full(p.size,np.nan)}
        def stats(self):return dict(success=True,return_status='injected_nonfinite')
    p.solver=NonfiniteSolver();a,log=p.command(np.zeros(4))
    assert log['fallback'] and log['qp_max_normalized_violation'] is None and a==0
    json.dumps(log,allow_nan=False)


def test_pending_violation_and_invalid_delay_are_visible(config_model):
    c,m=config_model;p=PredictiveController(m,envelope(m),.08,c['control']);p.queue=[1.,1.]
    _,l=p.command(np.array([0.,0.,.12,.3]));assert l['pending_queue_constraint_violation'] and l['fallback']
    with pytest.raises(ValueError):PredictiveController(m,envelope(m),.03,c['control'])


def test_event_schedule_and_independent_holdout(config_model):
    c,m=config_model;d=ScheduledDisturbance(.06,[[.32,-.14],[.88,.10]])
    jumps,force=d.arrays(m.human,60)
    assert jumps[0]==.06 and jumps[8]==-.14 and jumps[22]==.10 and np.count_nonzero(jumps)==3 and not force.any()
    with pytest.raises(ValueError):ScheduledDisturbance(0,[[.33,.1]]).arrays(m.human,60)
    plan=cases(c);assert len(plan)==768 and len({r['case_id'] for r in plan})==768
    assert plan==cases(c) # seeded draw independent of outcome and controller
    for s in c['test_scenarios']:
        assert s['initial_v_m_s']!=0 or any(t>0 for t,v in s['events'])


def test_endpoint_and_sustained_are_distinct(config_model):
    c,m=config_model;cc=c['_coupled'];t=np.arange(61)*.04;s=np.zeros((61,4))
    quiet=recovery_metrics(t,s,s,1.6,2.4,0,m,cc,True)
    assert quiet['terminal_recovered'] and quiet['sustained_recovered']
    s[50,0]=.02
    escaped=recovery_metrics(t,s,s,1.6,2.4,0,m,cc,True)
    assert escaped['terminal_recovered'] and not escaped['sustained_recovered']
    incomplete=recovery_metrics(t[:42],s[:42],s[:42],1.6,2.4,0,m,cc,True)
    assert incomplete['terminal_recovered'] and not incomplete['sustained_recovered']
    invalid=recovery_metrics(t,s,s,1.6,2.4,0,m,cc,False)
    assert not invalid['terminal_recovered'] and not invalid['sustained_recovered']


@pytest.fixture(scope='module')
def real_chart(config_model):
    from h1_capability.experiment import setup
    from h1_coupled.chart import SpatialChart
    c,m=config_model;cc=copy.deepcopy(c['_coupled']);cc['chart']['extent_m']=.008
    _,robot,checker=setup(cc['_capability']);record=next(r for r in json.loads(Path(cc['_capability']['candidates']).read_text()) if r['id']=='ik0036')
    chart=SpatialChart(robot,checker,record,cc,False).build()
    return cc,chart,chart.envelope(.005,cc['contact'],cc['robot']['torso_budget_nm'])


def test_real_fk_contact_environment_and_independent_replay(config_model,real_chart,tmp_path):
    from h1_coupled.validation import validate
    c,m=config_model;cc,chart,e=real_chart
    s=np.array([0,.01,0,0.]);tt,ss,hy,hv,f=actual_step(m,chart,s,.02)
    assert np.max(abs(hy-ss[:,2,None]))<1e-6 and np.max(abs(hv-ss[:,3,None]))<1e-6
    assert np.max(abs(f-(m.k*(hy-ss[:,0,None])+m.c*(hv-ss[:,1,None]))))==0
    scenario=dict(initial_v_m_s=.01,events=[])
    r=simulate(m,chart,e,scenario,'mpc',c['control'],cc,.32,.64)
    path=tmp_path/'check.npz';v=validate(m,chart,e,r['shadow'],r['acceleration'],r['disturbance'],.64,cc,path)
    assert v['valid']
    with np.load(path) as d:assert np.max(abs(d['independent_human_nodes']-r['state'][:,:2]))<1e-9


def test_selection_has_no_test_outcome_dependency():
    from h1_selection.scoring import select
    source=inspect.getsource(select)
    assert 'closed_loop' not in source and 'test_plan' not in source and 'case_summary' not in source


def test_resume_rejects_changed_choice_and_preserves_original_freeze(config_model,tmp_path):
    from h1_selection.evaluation import run
    c,_=config_model;freeze=tmp_path/'test_selection_freeze.json'
    original=json.dumps(dict(sha256='original selection checksum'))
    freeze.write_text(original);(tmp_path/'selection.json').write_text('{"methods": "tampered"}')
    with pytest.raises(ValueError,match='selection changed'):run(c,tmp_path,1)
    assert freeze.read_text()==original
