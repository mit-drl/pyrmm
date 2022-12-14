'''
Create SystemSetup for 1-D single integrator
'''
from __future__ import division

import copyreg
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import odeint
from functools import partial

from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
from pyrmm.dynamics.simple_integrators import ode_1d_double_integrator as ode_1d

LIDAR_RANGE = 1000.0    # max range of lidar observation [m]

class DoubleIntegrator1DSetup(SystemSetup):
    ''' Double integrator system in 1D with an obstacle(s)
    '''
    def __init__(self, pos_bounds, vel_bounds, acc_bounds, obst_bounds):
        '''
        Args:
            pos_bounds : [float, float]
                lower and upper bounds of position state [m]
            vel_bounds : [float, float]
                lower and upper bounds of velocity state [m/s]
            acc_bounds : [float, float]
                lower and upper bounds of acceleration control input [m/s/s]
            obst_bounds : [float, float]
                lower and upper bounds of obstacle space [m]
        '''

        assert pos_bounds[0] < pos_bounds[1]
        assert vel_bounds[0] < vel_bounds[1]
        assert acc_bounds[0] <= acc_bounds[1]
        assert obst_bounds[0] < obst_bounds[1]

        # save init args for re-creation of object
        self.pos_bounds = pos_bounds
        self.vel_bounds = vel_bounds
        self.acc_bounds = acc_bounds
        self.obst_bounds = obst_bounds

        # observation lidar range
        self.lidar_range = LIDAR_RANGE

        # create state space and set bounds
        state_space = ob.RealVectorStateSpace(2)
        sbounds = ob.RealVectorBounds(2)
        sbounds.setLow(0, pos_bounds[0])
        sbounds.setHigh(0, pos_bounds[1])
        sbounds.setLow(1, vel_bounds[0])
        sbounds.setHigh(1, vel_bounds[1])
        state_space.setBounds(sbounds)

        # create control space and set bounds
        control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=1)
        cbounds = ob.RealVectorBounds(1)
        cbounds.setLow(acc_bounds[0])
        cbounds.setHigh(acc_bounds[1])
        control_space.setBounds(cbounds)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)

        # create and set propagator class from ODEs
        propagator = DoubleIntegrator1DStatePropagator(spaceInformation=space_info)
        space_info.setStatePropagator(propagator)

        # create and set state validity checker
        validityChecker = ob.StateValidityCheckerFn(partial(self.isStateValid, space_info))
        space_info.setStateValidityChecker(validityChecker)

        # call parent init to create simple setup
        super().__init__(space_information=space_info)

    def __reduce__(self):
        ''' Function to enable re-creation of unpickable object

        Note: See comments about potential risks here
        https://stackoverflow.com/a/50308545/4055705
        '''
        return (DoubleIntegrator1DSetup, (self.pos_bounds, self.vel_bounds, self.acc_bounds, self.obst_bounds))

    def isStateValid(self, spaceInformation, state):
        ''' check if state is in collision with obstacle (state bounds used for sampling)
        Args:
            spaceInformation : ob.SpaceInformationPtr
                state space information as given by SimpleSetup.getSpaceInformation
            state : ob.State
                state to check for validity
        
        Returns:
            True if state not in collision with obstacle
        '''
        if state[0] >= self.obst_bounds[0] and state[0] <= self.obst_bounds[1]:
            return False
        else: 
            return True

    def isPathValid(self, path):
        '''check if path intersects obstacles using env-specific trajectory checker
        
        Args:
            path : oc.Path
                OMPL representation of path to be checked
        
        Returns:
            true if all path states and interpolated lines are valid (not colliding with obstacles)

        Notes:
            This overrides SystemSetup function that just looks at discrete states, not interpolated paths
        '''

        # convert path to array of states
        n_states = path.getStateCount()
        # np_traj = np.array([state_ompl_to_numpy(path.getState(i)) for i in range(n_states)])

        for i in range(n_states-1):
            if not self.space_info.isValid(path.getState(i)):
                return False
            elif not self.space_info.isValid(path.getState(i+1)):
                return False
            elif np.sign(self.obst_bounds[0] - path.getState(i)[0]) != \
                np.sign(self.obst_bounds[0] - path.getState(i+1)[0]):
                # consecutive states in the trajectory are on different sides of the obstacle's l
                # lower bound
                return False

        return True

    def observeState(self, state) -> ArrayLike:
        '''query observation from a particular state
        Args:
            state : ob.SE2State
                state from which to make observation
        Returns:
            observation : ArrayLike
                array giving observation values with index ordering
                0: left lidar range pointing along negative x-axis direction [m]
                1: right lidar range pointing along positive x-axis direction [m]
                2: signed velocity [m/s]
        '''
        assert self.lidar_range > 0.0

        obs = np.zeros(3)

        # encode velocity
        obs[2] = state[1]

        # check if current state in collision with obstacle
        if not self.space_info.isValid(state):
            # in collision with obstacle, leaves lidar readings as zeros
            obs[0] = 0.0
            obs[1] = 0.0
            return obs

        else:

            # check if to left of obstacle
            if state[0] < self.obst_bounds[0]:
                # left lidar is max-ranged
                obs[0] = self.lidar_range
                # right lidar range to obstacle
                obs[1] = min(self.obst_bounds[0] - state[0], self.lidar_range)

            elif state[0] > self.obst_bounds[1]:
                # left lidar range to obstacle
                obs[0] = min(state[0] - self.obst_bounds[1], self.lidar_range)
                # right lidar is max-ranged
                obs[1] = self.lidar_range

            else:
                raise ValueError("State {} not expected to be valid".format(state))

            return obs

    def control_ompl_to_numpy(self, omplCtrl, npCtrl=None):
        """convert single integrator ompl control object to numpy array

        Args:
            omplCtrl : oc.Control
                1D double integrator control (i.e. acceleration) in ompl RealVectorControl format
            npCtrl : ndarray (1,)
                1D double integrator control represented in np array [acceleration]
                if not None, input argument is modified in place, else returned
        """
        ret = False
        if npCtrl is None:
            npCtrl = np.empty(1,)
            ret = True

        npCtrl[0] = omplCtrl[0]

        if ret:
            return npCtrl

class DoubleIntegrator1DStatePropagator(oc.StatePropagator):

    def __init__(self, spaceInformation):

        # Store information about space propagator operates on
        # NOTE: this serves the same purpose as the  protected attribute si_ 
        # but si_ does not seem to be accessible in python
        # Ref: https://ompl.kavrakilab.org/classompl_1_1control_1_1StatePropagator.html
        self.__si = spaceInformation
        super().__init__(si=spaceInformation)

    def propagate_path(self, state, control, duration, path):
        ''' propagate from start based on control, store path in-place
        Args:
            state : ob.State internal
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
            This function is similar, but disctinct from 'StatePropagator.propogate', thus its different name to no overload `propagate`. 
            propogate does not store or return the path to get to result
            
            Currently using scipy's odeint. This creates a dependency on scipy and is likely inefficient
            because it's perform the numerical integration in python instead of C++. 
            Could be improved later
        '''

        # unpack objects from space information for ease of use
        cspace = self.__si.getControlSpace()
        nctrldims = cspace.getDimension()
        cbounds = cspace.getBounds()
        nsteps = path.getStateCount()
        assert nsteps >= 2
        assert nctrldims == 1
        assert duration >= 0

        # package init state and time vector
        s0 = [state[0], state[1]]
        
        # create equi-spaced time vector based on number or elements
        # in path object
        t = np.linspace(0.0, duration, nsteps)

        # clip the control to ensure it is within the control bounds
        bounded_control = [np.clip(control[i], cbounds.low[i], cbounds.high[i]) for i in range(nctrldims)]

        # call scipy's ode integrator
        sol = odeint(ode_1d, s0, t, args=(bounded_control,))

        # store each intermediate point in the solution as pat of the path
        pstates = path.getStates()
        pcontrols = path.getControls()
        ptimes = path.getControlDurations()
        assert len(pcontrols) == len(ptimes) == nsteps-1
        for i in range(nsteps-1):
            pstates[i][0] = sol[i,0]
            pstates[i][1] = sol[i,1]
            for j in range(nctrldims):
                pcontrols[i][j] = bounded_control[j]
            ptimes[i] = t[i+1] - t[i]
        
        # store final state
        pstates[-1][0] = sol[-1,0]
        pstates[-1][1] = sol[-1,1]

    def canPropagateBackwards(self):
        return False

    def steer(self, from_state, to_state, control, duration):
        return False

    def canSteer(self):
        return False

_DUMMY_REALVECTOR2SPACE = ob.RealVectorStateSpace(2)

def _pickle_RealVectorStateSpace2(state):
    '''pickle OMPL RealVectorStateSpace2 object'''
    x = state[0]
    v = state[1]
    return _unpickle_RealVectorStateSpace2, (x, v)

def _unpickle_RealVectorStateSpace2(x, v):
    '''unpickle OMPL RealVectorStateSpace2 object'''
    state = _DUMMY_REALVECTOR2SPACE.allocState()
    state[0] = x
    state[1] = v
    return state

def update_pickler_RealVectorStateSpace2():
    '''updates pickler to enable pickling and unpickling of ompl objects'''
    copyreg.pickle(_DUMMY_REALVECTOR2SPACE.allocState().__class__, _pickle_RealVectorStateSpace2, _unpickle_RealVectorStateSpace2)

# def state_ompl_to_numpy(omplState, np_state=None):
#     """convert 1D double integrator ompl state to numpy array

#     Args:
#         omplState : ob.CompoundState
#             dubins 4d state in ompl CompoundState format
#         np_state : ndarray (4,)
#             dubins 4d state represented in np array in [x,y,theta,v] ordering
#     """
#     ret = False
#     if np_state is None:
#         np_state = np.empty(2,)
#         ret = True
#     np_state[0] = omplState[0]
#     np_state[1] = omplState[1]

#     if ret:
#         return np_state