from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import recoverability.solver as sol
from recoverability.actuator import Actuator
from recoverability.disturbance import Disturbance
from recoverability.feedback import simulate_feedback
from recoverability.metrics import validate
from recoverability.model import Model


def test_oracle_zero_and_analytic_impossibility():
    m=Model();a=Actuator()
    zero=sol.solve_oracle(m,a,Disturbance(),2)
    assert zero.status=='RECOVERABLE'
    np.testing.assert_allclose(zero.state,0,atol=1e-10)
    # DCM capture bound for any |f|<=F is F/(m*omega) < .8; finite response cannot improve it.
    assert .8 > a.force_max_N/(m.mass_kg*m.omega)
    # Even instantaneous maximum opposing force cannot bring the divergent component
    # into the terminal box by H=2 s (a necessary finite-horizon condition).
    w=m.omega; growth=np.exp(w*2)
    xi_lower=(.8/w)*growth-a.force_max_N/(m.mass_kg*w*w)*(growth-1)
    assert xi_lower > m.x_terminal_m+m.v_terminal_m_s/w
    impossible=sol.solve_oracle(m,a,Disturbance(delta_v_m_s=.8),2)
    assert impossible.status=='INFEASIBLE' and len(impossible.state)==0


@pytest.mark.parametrize('velocity',[.04,.2,.4,.64])
def test_symmetric_disturbance_classification_and_mirrored_paths(velocity):
    m=Model();a=Actuator()
    left=sol.solve_oracle(m,a,Disturbance(delta_v_m_s=velocity),1.2)
    right=sol.solve_oracle(m,a,Disturbance(delta_v_m_s=-velocity),1.2)
    assert left.status==right.status!='SOLVER_ERROR'
    if left.status=='RECOVERABLE':
        # L1 optima can be nonunique; compare reflected feasibility, not arbitrary optimizer choices.
        mirrored=validate(m,a,Disturbance(delta_v_m_s=-velocity),-left.state,-left.force,-left.command)
        assert mirrored['valid_path_dynamics'] and mirrored['terminal_met']
        assert left.metrics['force_effort_N_s']==pytest.approx(right.metrics['force_effort_N_s'],abs=1e-5)


@pytest.mark.parametrize('velocity',[.08,.2,.36,.48,.64])
def test_force_rate_delay_horizon_consistency(velocity):
    m=Model();a=Actuator(rate_N_s=300,delay_s=.1)
    d=Disturbance(delta_v_m_s=velocity)
    original=sol.solve_oracle(m,a,d,1.2)
    for stronger in (replace(a,force_max_N=200,command_max_N=200),replace(a,rate_N_s=1500),replace(a,delay_s=0)):
        r=sol.solve_oracle(m,stronger,d,1.2)
        assert r.status!='SOLVER_ERROR'
        if original.status=='RECOVERABLE':assert r.status=='RECOVERABLE'
    # Empirical regression on this grid; box terminal set does NOT imply a general horizon theorem.
    longer=sol.solve_oracle(m,a,d,2)
    if original.status=='RECOVERABLE':assert longer.status=='RECOVERABLE'


def test_oracle_posterior_and_tampering():
    m=Model();a=Actuator(delay_s=.1)
    d=Disturbance(delta_v_m_s=.2)
    r=sol.solve_oracle(m,a,d,2)
    assert r.status=='RECOVERABLE'
    assert r.metrics['valid_path_dynamics'] and r.metrics['terminal_met']
    np.testing.assert_allclose(r.force[:6],0,atol=1e-9)
    bad=r.force.copy();bad[2]=1
    assert not validate(m,a,d,r.state,bad,r.command)['valid_path_dynamics']
    bad=r.state.copy();bad[5,0]+=1e-3
    assert not validate(m,a,d,bad,r.force,r.command)['valid_path_dynamics']
    bad[1,0]=np.nan
    assert validate(m,a,d,bad,r.force,r.command)['nonfinite']


def test_future_disturbance_nonanticipation():
    m=Model();a=Actuator(delay_s=.1)
    zero=simulate_feedback(m,a,Disturbance(),2)
    plus=simulate_feedback(m,a,Disturbance(onset_s=.4,delta_v_m_s=.3),2)
    minus=simulate_feedback(m,a,Disturbance(onset_s=.4,delta_v_m_s=-.3),2)
    for r in (plus,minus):
        np.testing.assert_array_equal(r.command[:20],zero.command[:20])
        np.testing.assert_array_equal(r.force[:26],0)
    np.testing.assert_allclose(plus.state,-minus.state,atol=1e-12)


def test_force_pulse_feedback_sees_no_future_inputs():
    m=Model();a=Actuator()
    d=Disturbance(kind='force_pulse',onset_s=.2,force_N=100,duration_s=.2)
    r=simulate_feedback(m,a,d,2)
    np.testing.assert_array_equal(r.command[:11],0) # force first affects measured state at next node
    assert len(r.state)==101
    assert sol.solve_oracle(m,a,d,2).status=='RECOVERABLE'


@pytest.mark.parametrize('status',[1,3,4])
def test_solver_failure_not_infeasibility_or_success(monkeypatch,status):
    monkeypatch.setattr(sol,'linprog',lambda *a,**k:SimpleNamespace(status=status,message='injected diagnostic',x=None))
    r=sol.solve_oracle(Model(),Actuator(),Disturbance(),1)
    assert r.status=='SOLVER_ERROR' and r.solver_status==status


def test_solver_exception_recorded(monkeypatch):
    def broken(*args,**kwargs):raise RuntimeError('injected numerical failure')
    monkeypatch.setattr(sol,'linprog',broken)
    r=sol.solve_oracle(Model(),Actuator(),Disturbance(),1)
    assert r.status=='SOLVER_ERROR' and 'RuntimeError' in r.reason


def test_false_solver_success_rejected(monkeypatch):
    def corrupt(c,**kwargs):return SimpleNamespace(status=0,message='injected NaN',x=np.full(len(c),np.nan))
    monkeypatch.setattr(sol,'linprog',corrupt)
    r=sol.solve_oracle(Model(),Actuator(),Disturbance(),1)
    assert r.status=='SOLVER_ERROR' and r.metrics['nonfinite']


def test_initial_outside_path_is_not_silently_accepted():
    r=sol.solve_oracle(Model(),Actuator(),Disturbance(),1,initial=(.21,0))
    assert r.status=='INFEASIBLE'


def test_feedback_failure_is_not_infeasibility_certificate():
    r=simulate_feedback(Model(),Actuator(),Disturbance(delta_v_m_s=.8),2)
    assert r.status=='NOT_RECOVERED'


def test_post_validation_rejection_is_unknown_not_infeasible(monkeypatch):
    monkeypatch.setattr(sol,'validate',lambda *a,**k:{'valid_path_dynamics':False,'terminal_met':True})
    r=sol.solve_oracle(Model(),Actuator(),Disturbance(),1)
    assert r.status=='SOLVER_ERROR' and 'posterior' in r.reason
