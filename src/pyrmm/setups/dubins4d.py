'''
Create SystemSetup for Dubins-4D car, i.e. 
state : [x, y, heading, speed]
control : [d_theta, d_speed]
'''

import numpy as np

import pyrmm.dynamics.dubins4d as D4DD

from functools import partial
from scipy.integrate import odeint

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
        sbounds = dict()
        sbounds['xpos_low'] = float(env.state_space.low[0])
        sbounds['xpos_high'] = float(env.state_space.high[0])
        sbounds['ypos_low'] = float(env.state_space.low[1])
        sbounds['ypos_high'] = float(env.state_space.high[1])
        sbounds['speed_low'] = float(env.state_space.low[3])
        sbounds['speed_high'] = float(env.state_space.high[3])
        state_space = D4DD.Dubins4DStateSpace(bounds=sbounds)

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

        # create and set state validity checker
        validityChecker = ob.StateValidityCheckerFn(partial(self.isStateValid, space_info, env))
        space_info.setStateValidityChecker(validityChecker)

        # # call parent init to create simple setup
        # super().__init__(space_information=space_info)

    def isStateValid(self, spaceInformation, environment, state):
        ''' check ppm image colors for obstacle collision
        Args:
            spaceInformation : ob.SpaceInformationPtr
                state space information as given by SimpleSetup.getSpaceInformation
            environment : gym.Env
                envrionment defining obstacle set
            state : ob.State
                state to check for validity
        
        Returns:
            True if state in bound and not in collision with obstacles
        '''

        # convert state to numpy

        # check obstacle collision

        # check speed bounds
        raise NotImplementedError()


        
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
        s0 = np.empty(4,)
        state_ompl_to_numpy(omplState=state, np_state=s0)
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
        sol = odeint(D4DD.ode_dubins4d, s0, t, args=(bounded_control, speed_bounds))

        # store solution in result
        state_numpy_to_ompl(np_state=sol[-1,:], omplState=result)
        # result[0][0] = sol[-1,0]
        # result[0][1] = sol[-1,1]
        # result[1].value = sol[-1,2]
        # result[2][0] = sol[-1,3]

    def propagate_path(self, state, control, duration, path):
        ''' propagate from start based on control, store final state in result, store path to result
        Args:
            state : ob.State
                start state of propagation
            control : oc.Control
                control to apply during propagation
            duration : float
                duration of propagation
            path : oc.ControlPath
                path from state to result in nsteps. initial state is state, final state is result

        Returns:
            None


        Notes:
            This function is similar, but disctinct from 'StatePropagator.propagate', thus its different name to no overload `propagate`. 
            propagate does not store or return the path to get to result
            
            Currently using scipy's odeint. This creates a dependency on scipy and is likely inefficient
            because it's perform the numerical integration in python instead of C++. 
            Could be improved later
        '''
        raise NotImplementedError()

    def canPropagateBackwards(self):
        return False

    def steer(self, from_state, to_state, control, duration):
        return False

    def canSteer(self):
        return False

def state_ompl_to_numpy(omplState, np_state):
    """convert Dubins4d ompl state to numpy array

    Args:
        omplState : ob.CompoundState
            dubins 4d state in ompl CompoundState format
        np_state : ndarray (4,)
            dubins 4d state represented in np array in [x,y,theta,v] ordering
    """
    np_state[0] = omplState[0][0]
    np_state[1] = omplState[0][1]
    np_state[2] = omplState[1].value
    np_state[3] = omplState[2][0]

def state_numpy_to_ompl(np_state, omplState):
    """convert dubins4d state in numpy array in [x,y,theta,v] to ompl compound state

    Args:
        np_state : ndarray (4,)
            dubins 4d state represented in np array in [x,y,theta,v] ordering
        omplState : ob.CompoundState
            dubins 4d state in ompl CompoundState format
    """
    omplState[0][0] = np_state[0]
    omplState[0][1] = np_state[1]
    omplState[1].value = np_state[2]
    omplState[2][0] = np_state[3]
