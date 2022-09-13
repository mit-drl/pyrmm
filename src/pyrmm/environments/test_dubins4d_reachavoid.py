# Pytests for dubins4d_reachavoid environment
import pytest
import time
import numpy as np

from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv, CircleRegion
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
    d = [0,0,0,0]

    # ~~ ACT ~~
    sol = odeint(env._Dubins4dReachAvoidEnv__ode_dubins4d_truth, s0, t, args=(u,d))

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
    d = [0,0,0,0]

    # ~~ ACT ~~
    sol = odeint(env._Dubins4dReachAvoidEnv__ode_dubins4d_truth, s0, t, args=(u,d))

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
    d = [0,0,0,0]

    # ~~ ACT ~~
    sol = odeint(env._Dubins4dReachAvoidEnv__ode_dubins4d_truth, s0, t, args=(u,d))

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
    env.reset()
    t_start = time.time()

    # specify initial state and control
    s0 = np.array([0, 0, 0, 1])
    env._Dubins4dReachAvoidEnv__state = s0  # (x [m], y [m], theta [rad], v [m/s])
    u = [0, 0]

    # ~~ ACT ~~
    # wait a fixed amount of time and then propagate system
    time.sleep(0.3434317)
    t_elapsed = time.time() - t_start
    env._propagate_realtime_system(ctrl=u)

    # ~~ ASSERT ~~
    exp_x = s0[3] * np.cos(s0[2]) * (t_elapsed + env._base_sim_time_step)
    exp_y = s0[3] * np.sin(s0[2]) * (t_elapsed + env._base_sim_time_step)
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
                target=[goal.xc, goal.yc],
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
    env._goal = goal

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

def test_step_to_now_inactive_ctrl_0(dubins4d_reachavoid_env_undisturbed):
    '''Propagate system with step-to-now function and inactive control (thus CLF "driver")'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    if dubins4d_reachavoid_env_undisturbed is None:
        dubins4d_reachavoid_env_undisturbed = get_dubins4d_reachavoid_env_undisturbed()
    env = dubins4d_reachavoid_env_undisturbed

    # specify initial state and goal
    s0 = np.array([0, 0, 0, 0])
    goal = CircleRegion(5, 0, 1)

    # reset environment to capture precise timing
    env.reset()
    t_start = time.time()
    env._Dubins4dReachAvoidEnv__state = s0  # (x [m], y [m], theta [rad], v [m/s])
    env._goal = goal

    # ~~ ACT ~~
    # wait a fixed amount of time and then propagate system
    time.sleep(0.53599)
    t_elapsed = time.time() - t_start
    env.step_to_now(env.action_space.sample())

    # ~~ ASSERT ~~
    # exp_x = s0[3] * np.cos(s0[2]) * t_elapsed
    exp_y = s0[1]
    exp_theta = s0[2]
    # exp_v = s0[3]
    assert np.greater(env._Dubins4dReachAvoidEnv__state[0], 0.0)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[1], exp_y)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[2], exp_theta)
    assert np.greater(env._Dubins4dReachAvoidEnv__state[3], 0.0)

def test_step_to_now_active_ctrl_0(dubins4d_reachavoid_env_undisturbed):
    '''Propagate system with step-to-now function with a specified action'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    if dubins4d_reachavoid_env_undisturbed is None:
        dubins4d_reachavoid_env_undisturbed = get_dubins4d_reachavoid_env_undisturbed()
    env = dubins4d_reachavoid_env_undisturbed

    # specify initial state and control
    s0 = np.array([0, 0, 0, 1])
    c0 = env.action_space.sample()
    c0['active_ctrl'] = True
    c0['turnrate_ctrl'][0] = 0
    c0['accel_ctrl'][0] = 0

    # reset environment to capture precise timing
    env.reset()
    t_start = time.time()
    env._Dubins4dReachAvoidEnv__state = s0  # (x [m], y [m], theta [rad], v [m/s])
    env._cur_action = c0

    # ~~ ACT ~~
    # wait a fixed amount of time and then propagate system
    time.sleep(0.580)
    t_elapsed = time.time() - t_start
    env.step_to_now(env.action_space.sample())

    # ~~ ASSERT ~~
    exp_x = s0[3] * np.cos(s0[2]) * (t_elapsed + env._base_sim_time_step)
    exp_y = s0[3] * np.sin(s0[2]) * (t_elapsed + env._base_sim_time_step)
    exp_theta = s0[2]
    exp_v = s0[3]
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[0], exp_x, rtol=1e-1)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[1], exp_y)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[2], exp_theta)
    assert np.isclose(env._Dubins4dReachAvoidEnv__state[3], exp_v)

def test_step_to_now_active_ctrl_rew_and_done_0(dubins4d_reachavoid_env_undisturbed):
    '''Propagate system with step-to-now function and check termination and reward'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    if dubins4d_reachavoid_env_undisturbed is None:
        dubins4d_reachavoid_env_undisturbed = get_dubins4d_reachavoid_env_undisturbed()
    env = dubins4d_reachavoid_env_undisturbed
    env._max_episode_sim_time = 0.5

    # specify initial state and control
    s0 = np.array([0, 0, 0, 2])
    c0 = env.action_space.sample()
    c0['active_ctrl'] = True
    c0['turnrate_ctrl'][0] = 0
    c0['accel_ctrl'][0] = 0

    # sequence of goals, obstacles, and timings to be tested
    goals = [CircleRegion(2, 0, 1.5), CircleRegion(1, 0, 0.5), CircleRegion(2, 0, 0.5), CircleRegion(-1, 0, 0.99)]
    obstacles = [CircleRegion(2, 0, 1), CircleRegion(0.5, 0, 0.2), CircleRegion(3, 0, 0.5), CircleRegion(-2, 0, 0.5)]
    times = [0.350, 0.2, 0.1, 0.51]
    exp_rews = [1, -1, 0, 0]
    exp_dones = [True, True, False, True]

    for i in range(len(goals)):
        # reset environment to capture precise timing
        env.reset()
        env._Dubins4dReachAvoidEnv__state = s0  # (x [m], y [m], theta [rad], v [m/s])
        env._cur_action = c0
        env._goal = goals[i]
        env._obstacles[0] = obstacles[i]
        for j in range(1,env._n_obstacles):
            env._obstacles[j] = CircleRegion(100,100,1)

        # ~~ ACT ~~
        # wait a fixed amount of time and then propagate system
        time.sleep(times[i])
        _, rew, done, _ = env.step_to_now(env.action_space.sample())

        # ~~ ASSERT ~~
        assert rew == exp_rews[i]
        assert done == exp_dones[i]

def test_get_observation_obstacle_0():
    '''test observation ray casting'''

    # ~~ ARRANGE ~~
    # used for debugging purposes
    n_rays = 4
    ray_length = 5.0
    env = Dubins4dReachAvoidEnv(n_rays=n_rays, ray_length=ray_length)
    assert env.observation_space.shape == (5+n_rays,)

    # sequence of init states and obstacles to be tested
    states = [
        np.array([0, 0, 0, 1]),
        np.array([0, 0, -np.pi/2, 0]),
    ]
    obstacles = [
        CircleRegion(2, 0, 1),
        CircleRegion(2, 0, 1),
    ]
    exp_len_5 = [
        1.0,
        ray_length
    ]
    exp_len_6 = [
        ray_length,
        1.0
    ]

    for i in range(len(obstacles)):
        # reset environment to capture precise timing
        env.reset()
        env._Dubins4dReachAvoidEnv__state = states[i]  # (x [m], y [m], theta [rad], v [m/s])
        env._obstacles[0] = obstacles[i]
        for j,_ in enumerate(env._obstacles[1:]):
            # move other obstacles out of the way
            env._obstacles[j+1] = CircleRegion(100, 100, 1)

        # ~~ ACT ~~
        # wait a fixed amount of time and then propagate system
        obs = env._get_observation(state=env._Dubins4dReachAvoidEnv__state)

        # ~~ ASSERT ~~
        assert np.isclose(obs[5], exp_len_5[i])
        assert np.isclose(obs[6], exp_len_6[i])



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
    # test_propagate_realtime_system_undisturbed_inactive_ctrl(None)
    test_step_to_now_active_ctrl_rew_and_done_0(None)
    # test_get_observation_obstacle_0()
    # test_propagate_realtime_system_undisturbed_zero_ctrl(None)