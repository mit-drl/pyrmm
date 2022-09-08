'''
Create SystemSetup for Dubins-4D car, i.e. 
state : [x, y, heading, speed]
control : [d_theta, d_speed]
'''

import numpy as np

from scipy.integrate import odeint

from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
from pyrmm.dynamics.dubins4d import ode_dubins4d
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
        propagator = Dubins4DReachAvoidStatePropagator(spaceInformation=space_info)
        space_info.setStatePropagator(propagator)

        # # create and set state validity checker
        # validityChecker = TODO
        # space_info.setStateValidityChecker(validityChecker)

        # # call parent init to create simple setup
        # super().__init__(space_information=space_info)

        
class Dubins4DReachAvoidStatePropagator(oc.StatePropagator):
    def __init__(self, spaceInformation):
        '''
        spaceInformation : oc.SpaceInformation
            OMPL object containing information about state and control space
        '''
        # Store information about space propagator operates on
        # NOTE: this serves the same purpose asthe  protected attribute si_ 
        # but si_ does not seem to be accessible in python
        # Ref: https://ompl.kavrakilab.org/classompl_1_1control_1_1StatePropagator.html
        self.__si = spaceInformation
        super().__init__(si=spaceInformation)

    def propagate(self, state, control, duration, result):
        ''' propagate from start based on control, store in state
        Args:
            state : ob.State
                start state of propagation
            control : oc.Control
                control to apply during propagation
            duration : float
                duration of propagation
            result : ob.State
                end state of propagation, modified in place

        Notes:
            By default, propagate does not perform or is used in integration,
            even when defined through an ODESolver; see:
            https://ompl.kavrakilab.org/RigidBodyPlanningWithODESolverAndControls_8py_source.html
            https://ompl.kavrakilab.org/classompl_1_1control_1_1StatePropagator.html#a4bf54becfce458e1e8abfa4a37ae8dff
            Therefore we must implement an ODE solver ourselves.
            Currently using scipy's odeint. This creates a dependency on scipy and is likely inefficient
            because it's integrating in python instead of C++. 
            Could be improved later
        '''

        # package init state and time vector
        # NOTE: only using 2-step time vector. Not sure if this degrades 
        # accuracy or just reduces the amount of data output
        s0 = [state[0][0], state[0][1], state[1].value, state[2][0]]
        t = [0.0, duration]

        # clip the control to ensure it is within the control bounds
        cbounds = self.__si.getControlSpace().getBounds()
        bounded_control = [np.clip(control[i], cbounds.low[i], cbounds.high[i]) for i in range(2)]

        # formulate speed (state) bounds
        speed_bounds = [
            self.__si.getStateSpace().getSubspace(2).getBounds().low,
            self.__si.getStateSpace().getSubspace(2).getBounds().high,
        ]

        # call scipy's ode integrator
        sol = odeint(ode_dubins4d, s0, t, args=(bounded_control, speed_bounds))

        # store solution in result
        result[0][0] = sol[-1,0]
        result[0][1] = sol[-1,1]
        result[1].value = sol[-1,2]
        result[2][0] = sol[-1,3]

