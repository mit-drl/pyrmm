
import numpy as np
from hypothesis import given, strategies as st
from scipy.integrate import odeint
from pyrmm.dynamics.dubins import ode_dubins
from pyrmm.dynamics.simple_integrators import ode_1d_single_integrator as ode_1d

def test_ode_dubins_integration_0():
    '''test that the dubins ODE integrates to expected values'''

    # ~~~ ARRANGE ~~~
    # create initial conditions and time vector
    y0 = [300, 200, 0]
    t = [0, 10]

    # specify the control (turning rate) and speed 
    u = [0]
    speed = 1

    # ~~~ ACT ~~~
    sol = odeint(ode_dubins, y0, t, args=(u, speed))

    # ~~~ ASSERT ~~~
    # check that final timestep is as expected
    assert np.isclose(sol[-1,0], 310)
    assert np.isclose(sol[-1,1], 200)
    assert np.isclose(sol[-1,2], 0)

@given(
    x0 = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
    tf = st.floats(min_value=1e-9, max_value=1e9, allow_nan=False, allow_infinity=False),
    u = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False)    
)
def test_hypothesis_ode_1d_0(x0, tf, u):
    '''test that 1d single integrator integrates to expected value'''

    # ~~~ ARRANGE ~~~
    # ~~~ ACT ~~~
    sol = odeint(ode_1d, [x0], [0, tf], args=([u],))

    # ~~~ ASSERT ~~~
    # check that final timestep is as expected
    assert np.isclose(sol[-1,0], x0 + u*tf)