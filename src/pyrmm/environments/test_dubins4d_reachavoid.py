# Pytests for dubins4d_reachavoid environment
import pytest
import numpy as np

from .dubins4d_reachavoid import Dubins4dReachAvoidEnv
from scipy.integrate import odeint

def get_dubins4d_reachavoid_env_undisturbed():
    env =  Dubins4dReachAvoidEnv()

    # set disturbances to zero
    env._Dubins4dReachAvoidEnv__dist.ctrl.x_mean = 0.0
    env._Dubins4dReachAvoidEnv__dist.ctrl.x_std = 0.0
    env._Dubins4dReachAvoidEnv__dist.ctrl.y_mean = 0.0
    env._Dubins4dReachAvoidEnv__dist.ctrl.y_std = 0.0
    env._Dubins4dReachAvoidEnv__dist.ctrl.theta_mean = 0.0
    env._Dubins4dReachAvoidEnv__dist.ctrl.theta_std = 0.0
    env._Dubins4dReachAvoidEnv__dist.ctrl.v_mean = 0.0
    env._Dubins4dReachAvoidEnv__dist.ctrl.v_std = 0.0

    return env

@pytest.fixture
def dubins4d_reachavoid_env_undisturbed():
    return get_dubins4d_reachavoid_env_undisturbed()

def test_ode_dubins4d_truth_undisturbed_zero_ctrl(dubins4d_reachavoid_env_undisturbed):
    '''Propagate system ODE with no control or disturbances and ensure correct final state'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    if dubins4d_reachavoid_env_undisturbed is None:
        dubins4d_reachavoid_env_undisturbed = get_dubins4d_reachavoid_env_undisturbed()
    env = dubins4d_reachavoid_env_undisturbed

    # specify initial state and time and control
    s0 = [0, 0, 0, 1]   # (x [m], y [m], theta [rad], v [m/s])
    t = np.linspace(0, 1.0, 10, endpoint=True)
    u = [0, 0]

    # ~~ ACT ~~
    sol = odeint(env._Dubins4dReachAvoidEnv__ode_dubins4d_truth, s0, t, args=(u,))

    # ~~ ASSERT ~~
    assert np.isclose(sol[-1, 0], 1.0)
    assert np.isclose(sol[-1, 1], 0.0)
    assert np.isclose(sol[-1, 2], 0.0)
    assert np.isclose(sol[-1, 3], 1.0)

def test_ode_dubins4d_truth_undisturbed_ctrl_acc_1(dubins4d_reachavoid_env_undisturbed):
    '''Propagate system ODE with no disturbances and only accel control'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    if dubins4d_reachavoid_env_undisturbed is None:
        dubins4d_reachavoid_env_undisturbed = get_dubins4d_reachavoid_env_undisturbed()
    env = dubins4d_reachavoid_env_undisturbed

    # specify initial state and time and control
    s0 = [0, 0, 0, 1]   # (x [m], y [m], theta [rad], v [m/s])
    t = [0, 1]
    acc = 0.1
    u = [0, acc]

    # ~~ ACT ~~
    sol = odeint(env._Dubins4dReachAvoidEnv__ode_dubins4d_truth, s0, t, args=(u,))

    # ~~ ASSERT ~~
    exp_x = 0.5*acc*t[-1]**2 + s0[3]*t[-1] + s0[0]
    exp_v = acc*t[-1] + s0[3]
    assert np.isclose(sol[-1, 0], exp_x)
    assert np.isclose(sol[-1, 1], 0.0)
    assert np.isclose(sol[-1, 2], 0.0)
    assert np.isclose(sol[-1, 3], exp_v)

def test_ode_dubins4d_truth_undisturbed_ctrl_dtheta_1(dubins4d_reachavoid_env_undisturbed):
    '''Propagate system ODE with no disturbances and only turning control'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    if dubins4d_reachavoid_env_undisturbed is None:
        dubins4d_reachavoid_env_undisturbed = get_dubins4d_reachavoid_env_undisturbed()
    env = dubins4d_reachavoid_env_undisturbed

    # specify initial state and time and control
    s0 = [0, 0, 0, 1]   # (x [m], y [m], theta [rad], v [m/s])
    t = [0, 1]
    dtheta = np.pi/2
    u = [dtheta, 0]

    # ~~ ACT ~~
    sol = odeint(env._Dubins4dReachAvoidEnv__ode_dubins4d_truth, s0, t, args=(u,))

    # ~~ ASSERT ~~
    exp_x = 2/np.pi
    exp_y = 2/np.pi
    exp_theta = np.pi/2
    exp_v = s0[3]
    assert np.isclose(sol[-1, 0], exp_x)
    assert np.isclose(sol[-1, 1], exp_y)
    assert np.isclose(sol[-1, 2], exp_theta)
    assert np.isclose(sol[-1, 3], exp_v)


if __name__ == "__main__":
    test_ode_dubins4d_truth_undisturbed_zero_ctrl(None)