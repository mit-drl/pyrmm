
import numpy as np
from scipy.integrate import odeint
from pyrmm.dynamics.dubins import ode_dubins

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