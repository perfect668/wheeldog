from pathlib import Path
import copy
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from recoverability.model import Model
from recoverability.disturbance import Disturbance
from h1_coupled.contact import CoupledModel
from h1_coupled.solver import solve_offline,CausalFeedback,simulate_feedback
from h1_coupled.experiment import load_config,check_frozen


@pytest.fixture
def coupled():
    c=load_config('configs/m1_coupled.yaml')
    return c,CoupledModel(c['_human'],c['contact'],c['experiment']['dt_s'])


def envelope():
    return dict(y_lower_m=-.1,y_upper_m=.1,speed_m_s=.4,acceleration_m_s2=1.,
        per_hand_force_lower_n=-50.,per_hand_force_upper_n=50.)


def test_lipm_coefficient_vertical_force_and_zero_contact_matches_m0(coupled):
    c,m=coupled
    assert m.human.mass_kg==70 and m.human.height_m==1
    assert m.human.gravity_m_s2==pytest.approx(9.81-60/70)
    assert m.cap_y==pytest.approx(np.sqrt(80**2-30**2))
    M=m.M.copy();M[1]=[m.human.omega**2,0,0,0]
    m.M=M;m._cache={}
    answer=m.step([.01,-.03,0,0],0,12.)
    assert np.allclose(answer[:2],m.human.step([.01,-.03],0,0,12.),atol=1e-13)


def test_contact_force_sign_static_damping_power_and_independent_ode(coupled):
    c,m=coupled
    s=np.array([.03,.1,-.02,-.05])
    assert m.per_hand_force(s)==pytest.approx(250*(-.05)+15*(-.15))
    assert abs(m.energy_power_residual(s))<1e-12
    rng=np.random.default_rng(42)
    for s in rng.normal(0,.1,size=(30,4)):
        assert abs(m.energy_power_residual(s))<1e-12
    s=np.array([.01,.1,0.,0.]);a=.3;d=10.
    independent=solve_ivp(lambda t,z:m.M@z+m.u*a+m.d*d,(0,m.dt),s,rtol=1e-12,atol=1e-13)
    assert np.allclose(independent.y[:,-1],m.step(s,a,d),atol=2e-13)


def test_offline_zero_terminal_delay_and_contact_constraints(coupled):
    c,m=coupled;e=envelope();d=Disturbance(delta_v_m_s=.08)
    result=solve_offline(m,d,1.6,e,.08)
    assert result['status']=='LP_FEASIBLE'
    assert np.max(abs(result['state'][-1]))<1e-7
    assert np.max(abs(result['acceleration'][:2]))<1e-9
    assert max(abs(m.per_hand_force(result['state'])))<=50+1e-5
    assert np.max(abs(result['state'][:,2]))<=.1+1e-7
    for k,a in enumerate(result['acceleration']):
        assert np.allclose(m.step(result['state'][k],a),result['state'][k+1],atol=1e-8)


def test_restricted_infeasibility_is_not_reported_as_global(coupled):
    c,m=coupled;e=envelope();e['per_hand_force_lower_n']=-.01;e['per_hand_force_upper_n']=.01
    result=solve_offline(m,Disturbance(delta_v_m_s=.4),1.2,e,.04)
    assert result['status']=='RESTRICTED_LP_INFEASIBLE'
    assert 'state' not in result


def test_causal_nonanticipativity_with_future_event_and_own_queue(coupled):
    c,m=coupled;e=envelope()
    quiet=simulate_feedback(m,Disturbance(delta_v_m_s=0),1.2,e,.04,c['feedback']['poles_s_inv'])
    event=simulate_feedback(m,Disturbance(onset_s=.32,delta_v_m_s=.08),1.2,e,.04,c['feedback']['poles_s_inv'])
    assert np.array_equal(quiet['acceleration'][:8],event['acceleration'][:8])
    assert quiet['command_logs'][:8]==event['command_logs'][:8]
    assert np.max(abs(quiet['state']))==0
    assert quiet['status']=='SIMULATED'


def test_force_cap_and_parameter_validation(coupled):
    c,m=coupled
    bad=copy.deepcopy(c['contact']);bad['resultant_cap_per_hand_n']=29
    with pytest.raises(ValueError):CoupledModel(c['_human'],bad,.04)
    with pytest.raises(ValueError):solve_offline(m,Disturbance(),1.2,envelope(),.03)


@pytest.fixture(scope='module')
def real_chart():
    import json
    from h1_capability.experiment import setup
    from h1_coupled.chart import SpatialChart
    c=load_config('configs/m1_coupled.yaml');c['chart']['extent_m']=.006
    _,robot,checker=setup(c['_capability'])
    record=next(r for r in json.loads(Path(c['_capability']['candidates']).read_text()) if r['id']=='ik0036')
    chart=SpatialChart(robot,checker,record,c,False).build()
    return c,chart,chart.envelope(.005,c['contact'],c['robot']['torso_budget_nm'])


def test_real_spatial_chart_derivatives_and_sufficient_acceleration_bound(real_chart):
    c,chart,e=real_chart;ys=np.linspace(e['y_lower_m'],e['y_upper_m'],231)
    d1=chart.spline(ys,1);d2=chart.spline(ys,2)
    eps=1e-7
    assert np.max(abs((chart.spline(ys+eps)-chart.spline(ys-eps))/(2*eps)-d1))<1e-6
    assert np.all(abs(d1)<=np.array(e['derivative_abs_bound'])+1e-8)
    assert np.all(abs(d2)<=np.array(e['second_derivative_abs_bound'])+1e-8)
    # Worst signs of scalar speed/acceleration cannot exhaust joint limits.
    acceleration=abs(d1)*e['acceleration_m_s2']+abs(d2)*e['speed_m_s']**2
    assert np.max(acceleration/c['_profile']['joint_acceleration_rad_s2'])<=1+1e-9
    assert e['per_hand_force_upper_n']<=np.sqrt(80**2-30**2)+1e-9


def test_nonlinear_replay_rejects_corrupt_dynamics_and_force_cap(real_chart):
    from h1_coupled.validation import validate
    c,chart,e=real_chart;m=CoupledModel(c['_human'],c['contact'],.04)
    state=np.zeros((3,4));control=np.zeros(2)
    quiet=validate(m,chart,e,state,control,Disturbance(delta_v_m_s=0),.08,c)
    assert quiet['valid'] and quiet['terminal_met']
    state[-1,0]=.02
    bad=validate(m,chart,e,state,control,Disturbance(delta_v_m_s=0),.08,c)
    assert not bad['valid'] and 'coupled_dynamics' in bad['reasons']
    cc=copy.deepcopy(c);cc['contact']['resultant_cap_per_hand_n']=30.0001
    capped=CoupledModel(cc['_human'],cc['contact'],.04)
    state[:]=[.01,0,0,0]
    bad=validate(capped,chart,e,state,control,Disturbance(delta_v_m_s=0),.08,cc)
    assert not bad['valid'] and 'contact_resultant_cap' in bad['reasons']


def test_independent_joint_replay_zero_and_nonlinear_contact(real_chart):
    from h1_coupled.direct import replay
    c,chart,e=real_chart;m=CoupledModel(c['_human'],c['contact'],.04)
    state=np.zeros((3,30));control=np.zeros((2,14))
    quiet=replay(chart,m,state,control,0,1,c)
    assert quiet['valid'] and max(abs(np.array(quiet['terminal_human'])))<1e-14
    disturbed=replay(chart,m,state,control,.2,1,c)
    assert not disturbed['valid'] and 'human_terminal' in disturbed['reasons']
