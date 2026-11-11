from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from recoverability.model import Model
from recoverability.actuator import Actuator, ForceActuator
from recoverability.disturbance import Disturbance
from recoverability.metrics import validate


def test_zero_equilibrium():
    m=Model()
    s=np.zeros(2)
    for _ in range(100): s=m.step(s,0,0)
    np.testing.assert_array_equal(s,[0,0])


def test_unforced_analytic_impulse_and_sign():
    m=Model();v=.3; t=m.dt_s
    expected=[v*np.sinh(m.omega*t)/m.omega,v*np.cosh(m.omega*t)]
    np.testing.assert_allclose(m.step([0,v],0,0),expected,rtol=1e-13)
    assert m.step([0,0],100,100)[1]>0
    assert m.step([.01,0],0,0)[1]>0


def test_exact_ramp_flow_against_independent_ode():
    m=Model(dt_s=.05)
    state=np.array([.02,-.1]);f0,f1,d=-20,30,40
    expected=solve_ivp(lambda t,s:[s[1],m.omega**2*s[0]+(f0+(f1-f0)*t/m.dt_s+d)/m.mass_kg],
                       [0,m.dt_s],state,rtol=1e-12,atol=1e-13).y[:,-1]
    np.testing.assert_allclose(m.step(state,f0,f1,d),expected,atol=1e-12)
    np.testing.assert_allclose(m.at_time(state,f0,f1,d,m.dt_s),expected,atol=1e-12)


def test_force_units_and_static_equilibrium():
    m=Model();x=.1;f=-m.mass_kg*m.omega**2*x
    np.testing.assert_allclose(m.step([x,0],f,f),[x,0],atol=1e-14)
    # Doubling mass halves the velocity increment from the same external force.
    light=m.step([0,0],0,0,100)
    heavy=replace(m,mass_kg=2*m.mass_kg).step([0,0],0,0,100)
    np.testing.assert_allclose(light,2*heavy)


def test_force_pulse_not_conflated_with_velocity_jump():
    m=Model();n=100
    j,d=Disturbance(kind='force_pulse',onset_s=.2,force_N=126,duration_s=.2).arrays(m,n)
    assert np.count_nonzero(j)==0 and np.count_nonzero(d)==10
    assert np.sum(d)*m.dt_s/m.mass_kg == pytest.approx(.36)
    # Equal force impulse area is not identical LIPM trajectory to instantaneous delta-v.
    s=np.zeros(2)
    for value in d[:20]: s=m.step(s,0,0,value)
    assert not np.allclose(s,[0,.36])


@pytest.mark.parametrize('kwargs',[{'mass_kg':0},{'dt_s':float('nan')},{'height_m':-1}])
def test_invalid_model_parameters(kwargs):
    with pytest.raises(ValueError):Model(**kwargs)


@pytest.mark.parametrize('disturbance',[
    Disturbance(onset_s=.001),Disturbance(kind='unknown'),Disturbance(force_N=2),
    Disturbance(kind='force_pulse',force_N=10,duration_s=.2,delta_v_m_s=.1),
    Disturbance(kind='force_pulse',onset_s=1.9,force_N=10,duration_s=.2),
    Disturbance(delta_v_m_s=float('nan'))])
def test_invalid_disturbances(disturbance):
    with pytest.raises(ValueError):disturbance.arrays(Model(),100)


def test_grid_and_delay_validation():
    with pytest.raises(ValueError):Model().steps(.013)
    with pytest.raises(ValueError):Actuator(delay_s=.013).delay_steps(.02)
    with pytest.raises(ValueError):Actuator(rate_N_s=-1)
    with pytest.raises(ValueError):Actuator(command_max_N=200)


def test_queue_rate_lag_and_command_limits_with_reversals():
    p=Actuator(rate_N_s=200,delay_s=.1,response_tau_s=.16)
    a=ForceActuator(p,.02)
    fs=[0];us=[]
    for k in range(100):
        _,f,u=a.advance(1000 if k%30<15 else -1000)
        fs.append(f);us.append(u)
    fs,us=np.array(fs),np.array(us)
    assert np.max(np.abs(np.diff(fs))) <= 4+1e-12
    assert np.max(abs(fs))<=150 and np.max(abs(us))<=150
    np.testing.assert_array_equal(fs[:6],np.zeros(6))
    delivered=np.r_[np.zeros(5),us[:-5]]
    np.testing.assert_allclose(fs[1:],(1-a.alpha)*fs[:-1]+a.alpha*delivered,atol=1e-12)


def test_continuous_path_extrema_detect_node_hidden_violation():
    m=Model(dt_s=.1,x_safe_m=.02)
    # Concave arc starts and ends below .02 but peaks in the interval.
    initial=np.array([.019,.08]); f=-120
    end=m.step(initial,f,f)
    peaks=m.interval_extrema(initial,f,f,0)
    dense=np.array([m.at_time(initial,f,f,0,t) for t in np.linspace(0,.1,10001)])
    np.testing.assert_allclose(peaks,np.max(np.abs(dense),axis=0),atol=2e-10)
    assert max(initial[0],end[0]) < m.x_safe_m < peaks[0]


def test_extrema_random_independent_dense_sampling():
    rng=np.random.default_rng(74);m=Model(dt_s=.05)
    for _ in range(30):
        s=rng.uniform(-.2,.2,2);f0,f1,d=rng.uniform(-150,150,3)
        extrema=m.interval_extrema(s,f0,f1,d)
        dense=np.array([m.at_time(s,f0,f1,d,t) for t in np.linspace(0,.05,301)])
        assert np.all(extrema >= np.max(np.abs(dense),axis=0)-1e-10)
        assert np.max(extrema-np.max(np.abs(dense),axis=0))<2e-6
