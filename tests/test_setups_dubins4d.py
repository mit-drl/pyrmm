import pickle
import numpy as np

import pyrmm.dynamics.dubins4d as D4D

from ompl import base as ob
from ompl import control as oc
from copy import deepcopy

from pyrmm.setups.dubins4d import Dubins4dReachAvoidSetup, \
    Dubins4dReachAvoidStatePropagator, \
    state_ompl_to_numpy, state_numpy_to_ompl, \
    update_pickler_dubins4dstate
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv

def test_Dubins4DReachAvoidSetup_init_0():
    
    # ~~~ ARRANGE ~~~
    # create default environment
    env = Dubins4dReachAvoidEnv()

    # ~~~ ACT ~~~
    # create dubins4d reach avoid setup
    d4d_setup = Dubins4dReachAvoidSetup(env = env)

    # ~~~ ASSERT ~~~
    pass

# def test_Dubins4DReachAvoid_isStateValid_0():
#     """check that a speed-violating path is invalid"""
#     # ~~~ ARRANGE ~~~
#     env = Dubins4dReachAvoidEnv()
#     d4d_setup = Dubins4DReachAvoidSetup(env=env)
#     is_valid_fn = d4d_setup.space_info.getStateValidityChecker()
#     s0 = d4d_setup.space_info.allocState()
#     s0[2][0] = env.state_space.high[3] + 0.1

#     # ~~~ ACT ~~~
#     is_valid = is_valid_fn.isValid(s0)

#     # ~~~ ASSERT ~~~
#     assert not is_valid

def test_Dubins4DReachAvoid_isStateValid_1():
    """check that a obst-violating path is invalid"""
    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    d4d_setup = Dubins4dReachAvoidSetup(env=env)
    is_valid_fn = d4d_setup.space_info.getStateValidityChecker()
    s0 = d4d_setup.space_info.allocState()
    s0[0][0] = float(env._obstacles[0].xc)
    s0[0][1] = float(env._obstacles[0].yc)

    # ~~~ ACT ~~~
    is_valid = is_valid_fn.isValid(s0)

    # ~~~ ASSERT ~~~
    assert not is_valid

def test_Dubins4DReachAvoid_isStateValid_2():
    """check that state is valid"""
    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    d4d_setup = Dubins4dReachAvoidSetup(env=env)
    is_valid_fn = d4d_setup.space_info.getStateValidityChecker()
    s0 = d4d_setup.space_info.allocState()
    s0[0][0] = float(env.state_space.high[0] + 100.)
    s0[0][1] = float(env.state_space.high[1] + 100.)
    s0[2][0] = float(env.state_space.low[3])

    # ~~~ ACT ~~~
    is_valid = is_valid_fn.isValid(s0)

    # ~~~ ASSERT ~~~
    assert is_valid

def test_Dubins4DReachAvoid_isPathValid_0():
    """check that unobstructed path is valid"""
    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    for j in range(env._n_obstacles):
        env._obstacles[j].xc = 100
        env._obstacles[j].yc = 100
    d4d_setup = Dubins4dReachAvoidSetup(env=env)

    # create state
    np_s0 = np.array([-1, -1, np.pi/4, 1])
    np_s1 = np.array([1, 1, np.pi/4, 1])
    s0 = d4d_setup.space_info.allocState()
    s1 = d4d_setup.space_info.allocState()
    state_numpy_to_ompl(np_s0, s0)
    state_numpy_to_ompl(np_s1, s1)

    # create path
    pth = oc.PathControl(d4d_setup.space_info)
    pth.append(s0)
    pth.append(s1)

    # ~~~ ACT ~~~
    is_valid = d4d_setup.isPathValid(pth)

    # ~~~ ASSERT ~~~
    assert is_valid

def test_Dubins4DReachAvoid_isPathValid_1():
    """check that obstructed path is invalid"""
    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    env._obstacles[0].xc = 0
    env._obstacles[0].yc = 0
    for j in range(1,env._n_obstacles):
        env._obstacles[j].xc = 100
        env._obstacles[j].yc = 100
    d4d_setup = Dubins4dReachAvoidSetup(env=env)

    # create state
    np_s0 = np.array([-1, -1, np.pi/4, 1])
    np_s1 = np.array([1, 1, np.pi/4, 1])
    s0 = d4d_setup.space_info.allocState()
    s1 = d4d_setup.space_info.allocState()
    state_numpy_to_ompl(np_s0, s0)
    state_numpy_to_ompl(np_s1, s1)

    # create path
    pth = oc.PathControl(d4d_setup.space_info)
    pth.append(s0)
    pth.append(s1)

    # ~~~ ACT ~~~
    is_valid = d4d_setup.isPathValid(pth)

    # ~~~ ASSERT ~~~
    assert not is_valid

def test_Dubins4dReachAvoidSetup_observeState_0():
    """check simple state observation is as expected"""

    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    # env._obstacle.xc = 1.0
    # env._obstacle.yc = 0.0
    env._goal.xc = 0.0
    env._goal.yc = 1.0
    d4d_setup = Dubins4dReachAvoidSetup(env=env)

    # create state
    np_s0 = np.array([0.08063903, 0.3758181 , 0.06863028, 0.25325886])
    s0 = d4d_setup.space_info.allocState()
    state_numpy_to_ompl(np_s0, s0)

    # ~~~ ACT ~~~
    # observe state
    obs = d4d_setup.observeState(state=s0)

    # ~~~ ASSERT ~~~
    assert len(obs) == 17
    assert np.isclose(obs[0], 0)
    assert np.isclose(obs[1], -0.08063903)
    assert np.isclose(obs[2], 1-0.3758181)
    assert np.isclose(obs[3], 0.06863028)
    assert np.isclose(obs[4], 0.25325886)

def test_Dubins4dReachAvoidSetup_observeState_1():
    """check simple state observation is as expected"""

    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    env._obstacles[0].xc = 1.0
    env._obstacles[0].yc = 0.0
    for j in range(1,env._n_obstacles):
        env._obstacles[j].xc = 100
        env._obstacles[j].yc = 100
    obst_r = 0.5 + 1e-3
    env._obstacles[0].r = obst_r
    env._goal.xc = 0.0
    env._goal.yc = 1.0
    d4d_setup = Dubins4dReachAvoidSetup(env=env)

    # create state
    np_s0 = np.array([0., 0., 0., 0.])
    s0 = d4d_setup.space_info.allocState()
    state_numpy_to_ompl(np_s0, s0)

    # ~~~ ACT ~~~
    # observe state
    obs = d4d_setup.observeState(state=s0)

    # ~~~ ASSERT ~~~
    assert len(obs) == 17
    # sim time
    assert np.isclose(obs[0], 0)

    # goal relative position
    assert np.isclose(obs[1], 0)
    assert np.isclose(obs[2], 1)

    # absolute heading and velocity
    assert np.isclose(obs[3], 0)
    assert np.isclose(obs[4], 0)

    # ray cast
    assert np.isclose(obs[5], 1-obst_r)
    theta = np.pi/6
    alpha = np.pi - np.arcsin(1*np.sin(theta)/obst_r)
    beta = np.pi - theta - alpha
    b = obst_r * np.sin(beta)/np.sin(theta)
    assert np.isclose(obs[6], b, rtol=1e-2)
    assert np.allclose(obs[7:-1], 10.0)
    assert np.isclose(obs[-1], b, rtol=1e-2)

def test_Dubins4dReachAvoidSetup_observeState_2():
    """check observation from within an obstacle"""

    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    env._obstacles[0].xc = 0.0
    env._obstacles[0].yc = 0.0
    for j in range(1,env._n_obstacles):
        env._obstacles[j].xc = 100
        env._obstacles[j].yc = 100
    obst_r = 0.5 + 1e-3
    env._obstacles[0].r = obst_r
    env._goal.xc = 0.0
    env._goal.yc = 1.0
    d4d_setup = Dubins4dReachAvoidSetup(env=env)

    # create state
    np_s0 = np.array([0., 0., 0., 0.])
    s0 = d4d_setup.space_info.allocState()
    state_numpy_to_ompl(np_s0, s0)

    # ~~~ ACT ~~~
    # observe state
    obs = d4d_setup.observeState(state=s0)

    # ~~~ ASSERT ~~~
    assert len(obs) == 17
    # sim time
    assert np.isclose(obs[0], 0)

    # goal relative position
    assert np.isclose(obs[1], 0)
    assert np.isclose(obs[2], 1)

    # absolute heading and velocity
    assert np.isclose(obs[3], 0)
    assert np.isclose(obs[4], 0)

    # ray cast
    assert np.allclose(obs[5:], 0.0)

def test_Dubins4dReachAvoidSetup_stateSampler_0():
    """check that state sampler samples expected region and is not trivial in any dimension"""
    # ~~~ ARRANGE ~~~
    n_samples = 1024
    env = Dubins4dReachAvoidEnv()
    ds = Dubins4dReachAvoidSetup(env=env)
    sampler = ds.space_info.allocStateSampler()
    ssamples = n_samples * [None] 
    np_ssamples = n_samples * [None]

    # ~~~ ACT ~~~
    # draw samples
    for i in range(n_samples):
        ssamples[i] = ds.space_info.allocState()
        sampler.sampleUniform(ssamples[i])
        np_ssamples[i] = state_ompl_to_numpy(omplState=ssamples[i])

    np_ssamples = np.asarray(np_ssamples)
    sample_xmin = np.min(np_ssamples[:,0])
    sample_xmax = np.max(np_ssamples[:,0])
    sample_ymin = np.min(np_ssamples[:,1])
    sample_ymax = np.max(np_ssamples[:,1])
    sample_tmin = np.min(np_ssamples[:,2])
    sample_tmax = np.max(np_ssamples[:,2])
    sample_vmin = np.min(np_ssamples[:,3])
    sample_vmax = np.max(np_ssamples[:,3])

    # ~~~ ASSERT ~~~

    # check that sampling is in expected bounds
    assert np.greater(sample_xmin, env.state_space.low[0])
    assert np.less(sample_xmax, env.state_space.high[0])
    assert np.greater(sample_ymin, env.state_space.low[1])
    assert np.less(sample_ymax, env.state_space.high[1])
    assert np.greater(sample_tmin, env.state_space.low[2])
    assert np.less(sample_tmax, env.state_space.high[2])
    assert np.greater(sample_vmin, env.state_space.low[3])
    assert np.less(sample_vmax, env.state_space.high[3])

    # check sampling is not trivial
    assert not np.isclose(sample_xmin, sample_xmax, rtol=0.1)
    assert not np.isclose(sample_ymin, sample_ymax, rtol=0.1)
    assert not np.isclose(sample_tmin, sample_tmax, rtol=0.1)
    assert not np.isclose(sample_vmin, sample_vmax, rtol=0.1)

def test_Dubins4dReachAvoidSetup_estimateRiskMetric_zero_risk_region_0():
    '''check that, when sampled away from obstacles, always produces zero risk'''

    # ~~~ ARRANGE ~~~
    n_samples = 32
    near_dist = 10.0
    duration = 5.0
    branch_fact = 16
    tree_depth = 2
    n_steps = 2

    # create environment and move obstacle far away
    env = Dubins4dReachAvoidEnv()
    for j in range(env._n_obstacles):
        env._obstacles[j].xc = 100
        env._obstacles[j].yc = 100

    # create system setup
    ds = Dubins4dReachAvoidSetup(env=env)

    # create sampling point
    np_s_near = np.zeros((4,))
    s_near = ds.space_info.allocState()
    state_numpy_to_ompl(np_state=np_s_near, omplState=s_near)

    # create sampler
    sampler = ds.space_info.allocStateSampler()
    ssamples = n_samples * [None] 
    rmetrics = n_samples * [None]

    # ~~~ ACT ~~~
    # sample states in zero-risk region and compute risk metrics

    for i in range(n_samples):

        # assign state and sample
        ssamples[i] = ds.space_info.allocState()
        sampler.sampleUniformNear(ssamples[i], s_near, near_dist)

        # compute risk metric
        rmetrics[i], _, _ = ds.estimateRiskMetric(ssamples[i], None, duration, branch_fact, tree_depth, n_steps)

        # print("Debug: risk metric={} at state ({},{},{})".format(rmetrics[i], ssamples[i].getX(), ssamples[i].getY(), ssamples[i].getYaw()))

        assert np.isclose(rmetrics[i], 0.0), "non-zero risk metric of {} for state ({},{},{})".format(rmetrics[i], ssamples[i].getX(), ssamples[i].getY(), ssamples[i].getYaw())

def test_Dubins4dReachAvoidSetup_estimateRiskMetric_inevitable_region_0():
    '''check that, when sampled in region of inevitable collision, risk is 1.0'''

    # ~~~ ARRANGE ~~~
    duration = 5.0
    branch_fact = 16
    tree_depth = 4
    n_steps = 2


    # create environment and move obstacle far away
    env = Dubins4dReachAvoidEnv()
    env._obstacles[0].xc = 1.0
    env._obstacles[0].yc = 1.0
    env._obstacles[0].r = 1.2
    for j in range(1,env._n_obstacles):
        env._obstacles[j].xc = 100
        env._obstacles[j].yc = 100

    # create system setup
    ds = Dubins4dReachAvoidSetup(env=env)

    # create sampling point
    np_s0 = np.zeros((4,))
    np_s0[2] = np.pi/4  # aim at obstacle
    np_s0[3] = 2.0  # set high speed
    s0 = ds.space_info.allocState()
    state_numpy_to_ompl(np_state=np_s0, omplState=s0)

    # ~~~ ACT ~~~

    # compute risk metric at region of inevitable collision
    rmetric, _, _ = ds.estimateRiskMetric(s0, None, duration, branch_fact, tree_depth, n_steps)

    # ~~~ ASSERT ~~~

    assert np.isclose(rmetric, 1.0)


def test_Dubins4DReachAvoidStatePropagator_propagate_0():

    # ~~~ ARRANGE ~~~
    # create state and control space
    state_space = ob.CompoundStateSpace()
    state_space.addSubspace(ob.RealVectorStateSpace(2), 1.0)    # xy-position [m]
    state_space.addSubspace(ob.SO2StateSpace(), 1.0)            # heading (theta)   [rad]
    state_space.addSubspace(ob.RealVectorStateSpace(1), 1.0)    # linear speed (v)  [m/s]

    # set state space bounds inherited from environment
    pos_bounds = ob.RealVectorBounds(2)
    pos_bounds.setLow(0, -10)
    pos_bounds.setHigh(0, 10)
    pos_bounds.setLow(1, -10)
    pos_bounds.setHigh(1, 10)
    state_space.getSubspace(0).setBounds(pos_bounds)

    speed_bounds = ob.RealVectorBounds(1)
    speed_bounds.setLow(0, -2)
    speed_bounds.setHigh(0, 2)
    state_space.getSubspace(2).setBounds(speed_bounds)

    # create control space and set bounds inherited from environment
    control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
    cbounds = ob.RealVectorBounds(2)
    cbounds.setLow(0, -1)
    cbounds.setHigh(0, 1)
    cbounds.setLow(1, -1)
    cbounds.setHigh(1, 1)
    control_space.setBounds(cbounds)

    # create space information for state and control space
    space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)
    propagator = Dubins4dReachAvoidStatePropagator(spaceInformation=space_info)

    # create state and control to propagate
    state = state_space.allocState()
    result = state_space.allocState()
    state[0][0] = 0.0
    state[0][1] = 0.0
    state[1].value = 0.0
    state[2][0] = 1.0
    
    ctrl = control_space.allocControl()
    ctrl[0] = 0.0
    ctrl[1] = 0.0

    # ~~~ ACT ~~~
    propagator.propagate(state, ctrl, 1.0, result)

    # ~~~ ASSERT ~~~
    assert np.isclose(result[0][0], 1.0)
    assert np.isclose(result[0][1], 0.0)
    assert np.isclose(result[1].value, 0.0)
    assert np.isclose(result[2][0], 1.0)

def test_Dubins4dReachAvoidStatePropagator_propagate_path_0():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    # create state space
    sbounds = dict()
    sbounds['xpos_low'] = -10.0
    sbounds['xpos_high'] = 10.0
    sbounds['ypos_low'] = -10.0
    sbounds['ypos_high'] = 10.0
    sbounds['speed_low'] = -2.0
    sbounds['speed_high'] = 2.0
    state_space = D4D.Dubins4dStateSpace(bounds=sbounds)

    # create control space and set bounds inherited from environment
    control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
    cbounds = ob.RealVectorBounds(2)
    cbounds.setLow(0, -1)
    cbounds.setHigh(0, 1)
    cbounds.setLow(1, -1)
    cbounds.setHigh(1, 1)
    control_space.setBounds(cbounds)

    # create space information for state and control space
    si = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)
    propagator = Dubins4dReachAvoidStatePropagator(spaceInformation=si)

    # create initial state
    np_s0 = np.array([0.0, 0.0, 0.0, 1.0])
    s0 = state_space.allocState()
    state_numpy_to_ompl(np_state=np_s0, omplState=s0)

    # create control input and duration
    c0 = control_space.allocControl()
    c0[0] = 0.0
    c0[1] = 0.0
    duration = 1.0

    # create path object and alloc 2 states
    path = oc.PathControl(si)
    path.append(state=si.allocState(), control=si.allocControl(), duration=0)
    path.append(state=si.allocState())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate_path(s0, c0, duration, path)
    
    # ~~~ ASSERT ~~~
    assert control_space.getDimension() == 2
    assert path.getStateCount() == 2
    assert path.getControlCount() == 1
    assert np.isclose(path.getState(0)[0][0], 0.0)
    assert np.isclose(path.getState(0)[0][1], 0.0)
    assert np.isclose(path.getState(0)[1].value, 0.0)
    assert np.isclose(path.getState(0)[2][0], 1.0)
    assert np.isclose(path.getControl(0)[0], 0.0)
    assert np.isclose(path.getControl(0)[1], 0.0)
    assert np.isclose(path.getControlDuration(0), 1.0)
    assert np.isclose(path.getState(1)[0][0], 1.0)
    assert np.isclose(path.getState(1)[0][1], 0.0)
    assert np.isclose(path.getState(1)[1].value, 0.0)
    assert np.isclose(path.getState(1)[2][0], 1.0)

def test_state_ompl_to_numpy_0():
    """check if copying states does not modify them"""
    # ~~~ ARRANGE ~~~
    sspace = D4D.Dubins4dStateSpace()

    omplState = sspace.allocState()
    np_state_orig = np.array([0.43438265, 0.66847181, 0.38747802, 0.00861762])
    np_state_new = np.empty(4,)

    # ~~~ ACT ~~~
    state_numpy_to_ompl(np_state_orig, omplState)
    state_ompl_to_numpy(omplState, np_state_new)

    # ~~~ ASSERT ~~~
    assert np.allclose(np_state_new, np_state_orig)

def test_update_pickler_dubins4dstate_0():
    """check that dubins4d state can be pickled and upickled without change"""
    # ~~~ ARRANGE ~~~
    # set pre-specified random state
    np_state = np.array([-2.300, -1.537, -0.890, 0.070])
    omplState = D4D.Dubins4dStateSpace().allocState()
    state_numpy_to_ompl(np_state=np_state, omplState=omplState)

    # ~~~ ACT ~~~
    # update pickler
    update_pickler_dubins4dstate()

    # pickle and unpicle ompl state
    omplState_copy = pickle.loads(pickle.dumps(omplState))

    # ~~~ ASSERT ~~~
    assert np.isclose(omplState[0][0], omplState_copy[0][0])
    assert np.isclose(omplState[0][1], omplState_copy[0][1])
    assert np.isclose(omplState[1].value, omplState_copy[1].value)
    assert np.isclose(omplState[2][0], omplState_copy[2][0])

def test_Dubins4dReachAvoidSetup_reduce_0():
    """check that pickling-unpickling setup object with modified env"""
    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    env._obstacles[0].xc = -51.388
    ds = Dubins4dReachAvoidSetup(env=env)

    # ~~~ ACT ~~~
    # modify env object to see if reproduced in setup object
    env._obstacles[0].xc = 79.476

    # pickle and unpickle setup object
    ds_copy = pickle.loads(pickle.dumps(ds))

    # ~~~ ASSERT ~~~
    # check that obstacle is in modified location is setup copy
    assert np.isclose(ds_copy.env._obstacles[0].xc, 79.476)

def test_Dubins4dReachAvoidSetup_control_ompl_to_numpy_0():
    """check that ompl control object is properly converted to numpy"""
    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    ds = Dubins4dReachAvoidSetup(env=env)

    # specify control object
    c = ds.space_info.allocControl()
    c[0] = 0.5654
    c[1] = 0.5224

    # ~~~ ACT ~~~
    # convert to numpy
    np_c = ds.control_ompl_to_numpy(omplCtrl=c)

    # ~~~ ASSERT ~~~
    assert np.isclose(c[0], np_c[0])
    assert np.isclose(c[1], np_c[1])

    
if __name__ == "__main__":
    # test_Dubins4DReachAvoidStatePropagator_propagate_0()
    # test_Dubins4DReachAvoid_isPathValid_0()
    # test_Dubins4dReachAvoidStatePropagator_propagate_path_0()
    # test_Dubins4dReachAvoidSetup_observeState_1()
    # test_Dubins4dReachAvoidSetup_stateSampler_0()
    # test_Dubins4dReachAvoidSetup_estimateRiskMetric_inevitable_region_0()
    test_Dubins4dReachAvoidSetup_estimateRiskMetric_zero_risk_region_0()