import numpy as np

import pyrmm.dynamics.dubins4d as D4DD

from ompl import base as ob
from ompl import control as oc
from copy import deepcopy

from pyrmm.setups.dubins4d import Dubins4DReachAvoidSetup, \
    Dubins4DReachAvoidStatePropagator, \
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
    d4d_setup = Dubins4DReachAvoidSetup(env = env)

    # ~~~ ASSERT ~~~
    pass

def test_Dubins4DReachAvoid_isStateValid_0():
    """check that a speed-violating path is invalid"""
    # ~~~ ARRANGE ~~~
    env = Dubins4dReachAvoidEnv()
    d4d_setup = Dubins4DReachAvoidSetup(env=env)
    is_valid_fn = d4d_setup.space_info.getStateValidityChecker()
    s0 = d4d_setup.space_info.allocState()
    s0[2][0] = env.state_space.high[3] + 0.1

    # ~~~ ACT ~~~
    is_valid = is_valid_fn.isValid(s0)

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
    propagator = Dubins4DReachAvoidStatePropagator(spaceInformation=space_info)

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

def test_state_ompl_to_numpy_0():
    """check if copying states does not modify them"""
    # ~~~ ARRANGE ~~~
    sspace = D4DD.Dubins4DStateSpace()

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