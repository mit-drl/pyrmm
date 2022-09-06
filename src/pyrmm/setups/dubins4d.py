'''
Create SystemSetup for Dubins-4D car, i.e. 
state : [x, y, heading, speed]
control : [d_theta, d_speed]
'''

from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
from pyrmm.environments.dubins4d_reachavoid import K_TURNRATE_CTRL, K_ACCEL_CTRL

class Dubins4DReachAvoidSetup(SystemSetup):
    def __init__(self, env):
        '''
        Args:
            env : Dubins4dReachAvoidEnv
                Instance of Dubins4dReachAvoidEnv
        '''

        # create state space
        state_space = ob.CompoundStateSpace()
        state_space.addSubspace(ob.RealVectorStateSpace(2), 1.0)    # xy-position [m]
        state_space.addSubspace(ob.SO2StateSpace(), 1.0)            # heading (theta)   [rad]
        state_space.addSubspace(ob.RealVectorStateSpace(1), 1.0)    # linear speed (v)  [m/s]

        # set state space bounds inherited from environment
        pos_bounds = ob.RealVectorBounds(2)
        pos_bounds.setLow(0, float(env.state_space.low[0]))
        pos_bounds.setHigh(0, float(env.state_space.high[0]))
        pos_bounds.setLow(1, float(env.state_space.low[1]))
        pos_bounds.setHigh(1, float(env.state_space.high[1]))
        state_space.getSubspace(0).setBounds(pos_bounds)

        speed_bounds = ob.RealVectorBounds(1)
        speed_bounds.setLow(0, float(env.state_space.low[3]))
        speed_bounds.setHigh(0, float(env.state_space.high[3]))
        state_space.getSubspace(2).setBounds(speed_bounds)

        # create control space and set bounds inherited from environment
        control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
        cbounds = ob.RealVectorBounds(2)
        cbounds.setLow(0, float(env.action_space[K_TURNRATE_CTRL].low[0]))
        cbounds.setHigh(0, float(env.action_space[K_TURNRATE_CTRL].high[0]))
        cbounds.setLow(1, float(env.action_space[K_ACCEL_CTRL].low[0]))
        cbounds.setHigh(1, float(env.action_space[K_ACCEL_CTRL].high[0]))
        control_space.setBounds(cbounds)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)

        # create and set propagator class from ODEs
        propagator = TODO
        space_info.setStatePropagator(propagator)

        # create and set state validity checker
        validityChecker = TODO
        space_info.setStateValidityChecker(validityChecker)

        # call parent init to create simple setup
        super().__init__(space_information=space_info)

        
