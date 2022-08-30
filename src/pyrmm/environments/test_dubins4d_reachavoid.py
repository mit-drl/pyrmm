# Pytests for dubins4d_reachavoid environment
import pytest
import time
import numpy as np

from .dubins4d_reachavoid import Dubins4dReachAvoidEnv, CircleRegion
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

def test_propagate_realtime_system_undisturbed_zero_ctrl(dubins4d_reachavoid_env_undisturbed):
    '''Propagate system with wall-clock time stepping with no disturbances and active-yet-zero control'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    if dubins4d_reachavoid_env_undisturbed is None:
        dubins4d_reachavoid_env_undisturbed = get_dubins4d_reachavoid_env_undisturbed()
    env = dubins4d_reachavoid_env_undisturbed

    # reset environment to capture precise timing
    t_start = time.time()
    env.reset()

    # specify initial state and control
    s0 = np.array([0, 0, 0, 1])
    env._Dubins4dReachAvoidEnv__state = s0  # (x [m], y [m], theta [rad], v [m/s])
    u = [0, 0]

    # ~~ ACT ~~
    # wait a fixed amount of time and then propagate system
    time.sleep(0.1434317)
    t_elapsed = time.time() - t_start
    env._propagate_realtime_system(ctrl=u)


    # ~~ ASSERT ~~
    exp_x = s0[3] * np.cos(s0[2]) * t_elapsed
    exp_y = s0[3] * np.sin(s0[2]) * t_elapsed
    exp_theta = s0[2]
    exp_v = s0[3]
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[0], exp_x, rtol=1e-2)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[1], exp_y)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[2], exp_theta)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[3], exp_v)

def test_solve_default_ctrl_clf_qp_0(dubins4d_reachavoid_env_undisturbed):
    '''solve inactive/default control lyapunov quadratic program and check control outputs'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    if dubins4d_reachavoid_env_undisturbed is None:
        dubins4d_reachavoid_env_undisturbed = get_dubins4d_reachavoid_env_undisturbed()
    env = dubins4d_reachavoid_env_undisturbed

    # specify initial state and goal and state constriants
    s0 = np.array([0, 0, 0, 0])
    goal = CircleRegion(5, 0, 1)
    v_min = 0.0
    v_max = 2.0

    # specify control constraints
    turnrate_min = -0.1
    turnrate_max = 0.1
    acc_min = -0.5
    acc_max = 0.5


    # ~~ ACT ~~
    # wait a fixed amount of time and then propagate system
    ctrl_n_del = env._solve_default_ctrl_clf_qp(
                state=s0,
                target=[goal.xc, goal.yc, 0.0],
                vmin=v_min, vmax=v_max,
                u1min=turnrate_min, u1max=turnrate_max,
                u2min=acc_min, u2max=acc_max,
                gamma_vmin=1, gamma_vmax=1,
                lambda_Vtheta=1, lambda_Vspeed=1,
                p_Vtheta=1, p_Vspeed=1
            )

    # ~~ ASSERT ~~
    assert np.isclose(ctrl_n_del[0], 0.0)   # turnrate control solution
    assert np.greater(ctrl_n_del[1], 0.0) # acceleration control solution
    # assert np.isclose(env._Dubins4dReachAvoidEnv__state[2], exp_theta)
    # assert np.greater(env._Dubins4dReachAvoidEnv__state[3], 0.0)

def test_propagate_realtime_system_undisturbed_inactive_ctrl(dubins4d_reachavoid_env_undisturbed):
    '''Propagate system with wall-clock time stepping with no disturbances and inactive control (thus controlled by passive CLF)'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    if dubins4d_reachavoid_env_undisturbed is None:
        dubins4d_reachavoid_env_undisturbed = get_dubins4d_reachavoid_env_undisturbed()
    env = dubins4d_reachavoid_env_undisturbed

    # specify initial state and goal
    s0 = np.array([0, 0, 0, 0])
    goal = CircleRegion(5, 0, 1)

    # reset environment to capture precise timing
    t_start = time.time()
    env.reset()
    env._Dubins4dReachAvoidEnv__state = s0  # (x [m], y [m], theta [rad], v [m/s])
    env.goal = goal

    # ~~ ACT ~~
    # wait a fixed amount of time and then propagate system
    time.sleep(0.5785567)
    t_elapsed = time.time() - t_start
    env._propagate_realtime_system(ctrl=None)

    # ~~ ASSERT ~~
    # exp_x = s0[3] * np.cos(s0[2]) * t_elapsed
    exp_y = s0[1]
    exp_theta = s0[2]
    # exp_v = s0[3]
    assert np.greater(env._Dubins4dReachAvoidEnv__state[0], 0.0)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[1], exp_y)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[2], exp_theta)
    assert np.greater(env._Dubins4dReachAvoidEnv__state[3], 0.0)

def test_check_traj_intersection_0():
    '''check that a known intersection is identified'''

    # ~~ ARRANGE ~~

    # Construct trajectory
    traj = np.array([
        [-0.5, 0, np.pi/4, 1],
        [10, 10, 0, 10]
    ])

    # create circular region
    circ = CircleRegion(0, 0, 1.0)

    # ~~ ACT ~~
    any_col, pt_col, edge_col = circ.check_traj_intersection(traj)

    # ~~ ASSERT ~~
    assert any_col
    assert pt_col[0]
    assert not pt_col[1]
    assert edge_col[0]

def test_check_traj_intersection_1():
    '''check that non-intersections are not mistakenly identified'''

    # ~~ ARRANGE ~~

    # Construct trajectory
    traj = np.array([
        [2, 0, np.pi/4, 1],
        [10, 10, 0, 10]
    ])

    # create circular region
    circ = CircleRegion(0, 0, 1.0)

    # ~~ ACT ~~
    any_col, pt_col, edge_col = circ.check_traj_intersection(traj)

    # ~~ ASSERT ~~
    assert not any_col
    assert not pt_col[0]
    assert not pt_col[1]
    assert not edge_col[0]

def test_check_traj_intersection_2():
    '''check longer, randomized trajectory'''

    # ~~ ARRANGE ~~

    # Construct trajectory
    traj = np.array([
        [-10, 1, *np.random.normal(size=2)],
        [-5.98247, -2, *np.random.normal(size=2)],
        [-2.5, -0.9, *np.random.normal(size=2)],
        [2.3, 1.0, *np.random.normal(size=2)],
        [10, 10, 0, 10]
    ])

    # create circular region
    circ = CircleRegion(0, 0, 2.0)

    # ~~ ACT ~~
    any_col, pt_col, edge_col = circ.check_traj_intersection(traj)

    # ~~ ASSERT ~~
    assert any_col
    assert not any(pt_col)
    assert not edge_col[0]
    assert not edge_col[1]
    assert not edge_col[3]
    assert edge_col[2]



if __name__ == "__main__":
    test_propagate_realtime_system_undisturbed_inactive_ctrl(None)