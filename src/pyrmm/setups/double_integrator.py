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

    observation_shape = (3,)

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
            state : ob.RealVectorStateSpace::StateType
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

    # def sample_control_numpy(self):
    #     """Randomly sample valid control in numpy format using numpy random

    #     Note: an error was found where the OMPL-provided control sampler
    #     produced repeated values when run in parallel processes using
    #     multiprocessing.Pool. No direct fix could be determined so
    #     this is a workaround that forces control sampling in numpy

    #     No equivalent bug has yet been identified in state sampling

    #     Args: 
    #         None

    #     Returns:
    #         np_ctrl : ArrayLike
    #             the ranomly sampled control in numpy-format (instead of ompl-format)
        
    #     """
    #     return np.random.uniform(*self.acc_bounds, (1,))

    def control_ompl_to_numpy(self, omplCtrl, npCtrl=None):
        """convert double integrator ompl control object to numpy array

        Args:
            omplCtrl : oc.Control
                1D double integrator control (i.e. acceleration) in ompl RealVectorControl format
            npCtrl : ndarray (1,)
                1D double integrator control represented in np array [acceleration]
                if not None, input argument is modified in place, else returned
        """

        # redirect to static method
        DoubleIntegrator1DSetup.control_ompl_to_numpy(omplCtrl=omplCtrl, npCtrl=npCtrl)

    @staticmethod
    def control_ompl_to_numpy(omplCtrl, npCtrl=None):
        """a static method to convert double integrator ompl control object to numpy array

        Note: this is static so that it can be called elsewhere (e.g. within StatePropagator
        class) without creating a dummy instance of the SystemSetup object.
        Note: at the same time, control_ompl_to_numpy is not made static itself because 
        we need to overload it at the parent class SystemSetup level 

        Args:
            omplCtrl : oc.Control
                1D double integrator control (i.e. acceleration) in ompl RealVectorControl format
            npCtrl : ArrayLike (1,)
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

    def control_numpy_to_ompl(self, npCtrl, omplCtrl):
        """convert double integrator control from numpy array to ompl control object in-place


        Args:
            npCtrl : ArrayLike (1,)
                1D double integrator control represented in np array [acceleration]
            omplCtrl : oc.Control
                1D double integrator control (i.e. acceleration) in ompl RealVectorControl format
        """

        # redirect to static method
        DoubleIntegrator1DSetup.control_numpy_to_ompl(omplCtrl=omplCtrl, npCtrl=npCtrl)

    @staticmethod
    def control_numpy_to_ompl(npCtrl, omplCtrl):
        """convert double integrator control from numpy array to ompl control object in-place


        Args:
            npCtrl : ArrayLike (1,)
                1D double integrator control represented in np array [acceleration]
            omplCtrl : oc.Control
                1D double integrator control (i.e. acceleration) in ompl RealVectorControl format
        """

        assert npCtrl.shape == (1,), "Unexpected shape {}".format(npCtrl.shape)
        omplCtrl[0] = npCtrl[0]

    @staticmethod
    def state_ompl_to_numpy(omplState, np_state=None):
        """convert 1D double integrator ompl state to numpy array

        Args:
            omplState : ob.State
                double integrator state in [pos, vel] ordering
            np_state : ArrayLike (2,)
                double integrator state represented in np array in [pos, vel] ordering
                if None, return np array, otherwise modify in place
        """
        ret = False
        if np_state is None:
            np_state = np.empty(2,)
            ret = True
        np_state[0] = omplState[0]
        np_state[1] = omplState[1]

        if ret:
            return np_state

    @staticmethod
    def state_numpy_to_ompl(np_state, omplState):
        """convert 1D double integrator state from numpy array to ompl object in-place

        Args:
            np_state : ArrayLike (2,)
                double integrator state represented in np array in [pos, vel] ordering
            omplState : ob.State
                double integrator state in [pos, vel] ordering
        """
        assert np_state.shape == (2,)
        omplState[0] = np_state[0]
        omplState[1] = np_state[1]

    @staticmethod
    def path_numpy_to_ompl(np_states: ArrayLike, np_controls: ArrayLike, np_times: ArrayLike, omplPath):
        """ Convert numpy-like description of a trajectory into ompl PathControl object (in place)

        TODO: This could probably be abstract up to the parent class, but would need to properly handle static methods
        (probably through re-directs in instance methods)

        Args:
            np_states : ArrayLike (n,2)
                array of n states in path in numpy-like format
            np_controls : ArrayLike (n-1,1)
                array of n-1 control inputs along path in numpy-like format
            np_times : ArrayLike (n,)
                array of n time steps of path in numpy-like format
            omplPath : oc.PathControl
                ompl PathControl object, modified in-place
        
        Returns:
            None
        """

        nsteps = len(np_states)
        assert nsteps == omplPath.getStateCount() == len(np_times)

        # store each intermediate point in the solution as pat of the path
        pstates = omplPath.getStates()
        pcontrols = omplPath.getControls()
        pdurs = omplPath.getControlDurations()
        assert len(pcontrols) == len(pdurs) == len(np_controls) == nsteps-1
        for i in range(nsteps-1):

            # write the state
            DoubleIntegrator1DSetup.state_numpy_to_ompl(np_state=np_states[i], omplState=pstates[i])

            # write the control 
            DoubleIntegrator1DSetup.control_numpy_to_ompl(npCtrl=np_controls[i], omplCtrl=pcontrols[i])

            # write the timing values
            pdurs[i] = np_times[i+1] - np_times[i]
        
        # store final state
        DoubleIntegrator1DSetup.state_numpy_to_ompl(np_state=np_states[-1], omplState=pstates[-1])

    @staticmethod
    def path_ompl_to_numpy(omplPath, np_states: ArrayLike, np_controls: ArrayLike, np_times: ArrayLike,):
        """ Convert ompl PathControl object to numpy arrays in-place

        TODO: This could probably be abstract up to the parent class, but would need to properly handle static methods
        (probably through re-directs in instance methods)

        Args:
            omplPath : oc.PathControl
                ompl PathControl object, modified in-place
            np_states : ArrayLike (n,2)
                array of n states in path in numpy-like format
            np_controls : ArrayLike (n-1,1)
                array of n-1 control inputs along path in numpy-like format
            np_times : ArrayLike (n,)
                array of n time steps of path in numpy-like format
        
        Returns:
            None (conversion performed in-place)
        """

        nsteps = len(np_states)
        assert nsteps == omplPath.getStateCount() == len(np_times)

        # store each intermediate point in the solution as pat of the path
        pstates = omplPath.getStates()
        pcontrols = omplPath.getControls()
        pdurs = omplPath.getControlDurations()
        assert len(pcontrols) == len(pdurs) == len(np_controls) == nsteps-1
        np_times[0] = 0.0
        for i in range(nsteps-1):

            # write the state
            DoubleIntegrator1DSetup.state_ompl_to_numpy(omplState=pstates[i], np_state=np_states[i])

            # write the control 
            DoubleIntegrator1DSetup.control_ompl_to_numpy(omplCtrl=pcontrols[i], npCtrl=np_controls[i])

            # write the timing values
            np_times[i+1] = pdurs[i] + np_times[i]
        
        # store final state
        DoubleIntegrator1DSetup.state_ompl_to_numpy(omplState=pstates[-1], np_state=np_states[-1])

class DoubleIntegrator1DStatePropagator(oc.StatePropagator):

    def __init__(self, spaceInformation):

        # Store information about space propagator operates on
        # NOTE: this serves the same purpose as the  protected attribute si_ 
        # but si_ does not seem to be accessible in python
        # Ref: https://ompl.kavrakilab.org/classompl_1_1control_1_1StatePropagator.html
        self.__si = spaceInformation
        super().__init__(si=spaceInformation)

    def propagate_path(self, state, control, duration, path):
        """ propagate from start based on control, store path in-place

        TODO: move this to parent class, should be possible if all children define
                a state_ompl_to_numpy and pathcontrol_numpy_to_ompl function

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
        """

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

        # expand bounded_control into array for conversion to ompl
        # PathControl object
        ctrl_array = np.array([bounded_control for i in range(nsteps-1)])

        DoubleIntegrator1DSetup.path_numpy_to_ompl(
            np_states=sol, 
            np_times=t, 
            np_controls=ctrl_array, 
            omplPath=path)

    def canPropagateBackwards(self):
        return False

    def steer(self, from_state, to_state, control, duration):
        return False

    def canSteer(self):
        return False

_DUMMY_REALVECTORSTATESPACE2 = ob.RealVectorStateSpace(2)
_DUMMY_REALVECTORCONTROLSPACE1 = oc.RealVectorControlSpace(stateSpace=_DUMMY_REALVECTORSTATESPACE2, dim=1)
_DUMMY_SPACEINFO = oc.SpaceInformation(stateSpace=_DUMMY_REALVECTORSTATESPACE2, controlSpace=_DUMMY_REALVECTORCONTROLSPACE1)

def _pickle_RealVectorStateSpace2(state):
    '''pickle OMPL RealVectorStateSpace2 object'''
    x = state[0]
    v = state[1]
    return _unpickle_RealVectorStateSpace2, (x, v)

def _unpickle_RealVectorStateSpace2(x, v):
    '''unpickle OMPL RealVectorStateSpace2 object'''
    state = _DUMMY_REALVECTORSTATESPACE2.allocState()
    state[0] = x
    state[1] = v
    return state

def update_pickler_RealVectorStateSpace2():
    '''updates pickler to enable pickling and unpickling of ompl objects'''
    copyreg.pickle(_DUMMY_REALVECTORSTATESPACE2.allocState().__class__, _pickle_RealVectorStateSpace2, _unpickle_RealVectorStateSpace2)

def _pickle_PathControl_DoubleIntegrator1D(path):
    # instatiate numpy arrays to later modify in place
    nsteps = path.getStateCount()
    np_states = np.empty((nsteps, 2))
    np_controls = np.empty((nsteps-1, 1))
    np_times = np.empty(nsteps)
    
    # convert ompl to numpy objects in place
    DoubleIntegrator1DSetup.path_ompl_to_numpy(omplPath=path, np_states=np_states, np_controls=np_controls, np_times=np_times)

    return _unpickle_PathControl_DoubleIntegrator1D, (np_states, np_controls, np_times)


def _unpickle_PathControl_DoubleIntegrator1D(np_states, np_controls, np_times):
    # instantiate ompl path object
    nsteps = len(np_states)
    path = oc.PathControl(_DUMMY_SPACEINFO)
    for j in range(nsteps-1):
        path.append(state=_DUMMY_SPACEINFO.allocState(), control=_DUMMY_SPACEINFO.allocControl(), duration=0)
    path.append(state=_DUMMY_SPACEINFO.allocState())

    # convert numpy values to omple object
    DoubleIntegrator1DSetup.path_numpy_to_ompl(np_states=np_states, np_controls=np_controls, np_times=np_times, omplPath=path)

    return path

def update_pickler_PathControl_DoubleIntegrator1D():
    '''updates pickler to enable pickling and unpickling of ompl PathControl objects'''
    copyreg.pickle(
        oc.PathControl(_DUMMY_SPACEINFO).__class__, 
        _pickle_PathControl_DoubleIntegrator1D, 
        _unpickle_PathControl_DoubleIntegrator1D)
