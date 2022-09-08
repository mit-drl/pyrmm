import numpy as np

from pyrmm.setups.dubins4d import Dubins4DReachAvoidSetup, Dubins4DReachAvoidStatePropagator
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
    
if __name__ == "__main__":
    test_Dubins4DReachAvoidStatePropagator_propagate_0()