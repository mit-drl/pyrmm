import numpy as np

import pyrmm.dynamics.dubins4d as D4DD

from ompl import base as ob
from ompl import control as oc
from copy import deepcopy

from pyrmm.setups.dubins4d import Dubins4dReachAvoidSetup, \
    Dubins4dReachAvoidStatePropagator, \
    state_ompl_to_numpy, state_numpy_to_ompl
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv
from ompl import base as ob
from ompl import control as oc
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
    s0[0][0] = float(env._obstacle.xc)
    s0[0][1] = float(env._obstacle.yc)

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
    env._obstacle.xc = 100
    env._obstacle.yc = 100
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
    env._obstacle.xc = 0
    env._obstacle.yc = 0
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
    state_space = D4DD.Dubins4dStateSpace(bounds=sbounds)

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
    sspace = D4DD.Dubins4dStateSpace()

    omplState = sspace.allocState()
    np_state_orig = np.array([0.43438265, 0.66847181, 0.38747802, 0.00861762])
    np_state_new = np.empty(4,)

    # ~~~ ACT ~~~
    state_numpy_to_ompl(np_state_orig, omplState)
    state_ompl_to_numpy(omplState, np_state_new)

    # ~~~ ASSERT ~~~
    assert np.allclose(np_state_new, np_state_orig)

    
if __name__ == "__main__":
    test_Dubins4DReachAvoidStatePropagator_propagate_0()
    # test_Dubins4DReachAvoid_isPathValid_0()
    # test_Dubins4dReachAvoidStatePropagator_propagate_path_0()