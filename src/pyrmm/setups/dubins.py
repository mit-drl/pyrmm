'''
Create SystemSetup for Dubins Car
'''
from __future__ import division

import numpy as np
import skimage
from scipy.integrate import odeint
from functools import partial

from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
from pyrmm.dynamics.dubins import ode_dubins
from pyrmm.utils.utils import is_pixel_free_space

class DubinsPPMSetup(SystemSetup):
    ''' Dubins car with ppm file for obstacle configuration space
    '''
    def __init__(self, ppm_file, speed, min_turn_radius, lidar_resolution=None, lidar_n_rays=None):
        '''
        Args:
            ppm_file : str
                file path to ppm image used as obstacle configuration space
            speed : float
                tangential speed of dubins vehicle
            min_turn_radius : float
                minimum turning radius of the dubins vehicle
            lidar_resolution : float
                step length of ray for validity checking (i.e. obstacle collision)
            lidar_n_rays : int
                number of lidar rays to be evenly spaced from 0 to 2pi
            
        '''

        assert speed > 0
        assert min_turn_radius > 0

        # save init args for re-creation of object
        self.ppm_file = ppm_file
        self.speed = speed
        self.min_turn_radius = min_turn_radius
        self.lidar_resolution = lidar_resolution
        self.lidar_n_rays = lidar_n_rays

        # generate configuration space from ppm file
        self.ppm_config_space = ou.PPM()
        self.ppm_config_space.loadFile(self.ppm_file)

        # compute lidar angles
        if lidar_resolution is not None and lidar_n_rays is not None:
            self.lidar_angles = np.linspace(0, 2*np.pi, num=lidar_n_rays, endpoint=False)
        else:
            self.lidar_angles = None

        # create state space and set bounds
        # state_space = ob.DubinsStateSpace(turningRadius=turning_radius)
        state_space = ob.SE2StateSpace()
        sbounds = ob.RealVectorBounds(2)
        sbounds.setLow(0, 0.0)
        sbounds.setHigh(0, self.ppm_config_space.getWidth())
        sbounds.setLow(1, 0.0)
        sbounds.setHigh(1, self.ppm_config_space.getHeight())
        state_space.setBounds(sbounds)

        # create control space and set bounds
        # Control space is turnrate in [rad/sec]
        control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=1)
        dtheta = self.speed/self.min_turn_radius
        cbounds = ob.RealVectorBounds(1)
        cbounds.setLow(-dtheta)
        cbounds.setHigh(dtheta)
        control_space.setBounds(cbounds)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)

        # create and set state validity checker
        validityChecker = ob.StateValidityCheckerFn(partial(self.isStateValid, space_info))
        space_info.setStateValidityChecker(validityChecker)

        # call parent init to create simple setup
        super().__init__(
            space_information=space_info,
            eom_ode=lambda y, t, u: ode_dubins(y, t, u, self.speed)
        )

    def __reduce__(self):
        ''' Function to enable re-creation of unpickable object

        Note: See comments about potential risks here
        https://stackoverflow.com/a/50308545/4055705
        '''
        return (DubinsPPMSetup, (self.ppm_file, self.speed, self.min_turn_radius))

    def isStateValid(self, spaceInformation, state):
        ''' check ppm image colors for obstacle collision
        Args:
            spaceInformation : ob.SpaceInformationPtr
                state space information as given by SimpleSetup.getSpaceInformation
            state : ob.State
                state to check for validity
        
        Returns:
            True if state in bound and not in collision with obstacles
        '''
        # if spaceInformation.satisfiesBounds(state):

        # check if state satisfies translational bounds (ignore rotational group)
        if self.space_info.getStateSpace().getSubspace(0).satisfiesBounds(state[0]):

            # if in state space bounds, check collision based
            # on ppm pixel color
            w = int(state.getX())
            h = int(state.getY())

            c = self.ppm_config_space.getPixel(h, w)
            return is_pixel_free_space(c)

        else:
            return False

    def isPathValid(self, path):
        '''check if path intersects obstacles in ppm image using bresenham lines
        
        Args:
            path : oc.Path
        
        Returns:
            true if all path states and interpolated bresenham lines are valid (not colliding with obstacles)

        Notes:
            This overrides SystemSetup function that just looks at discrete states, not interpolated paths
        '''
        for i in range(path.getStateCount()-1):

            # Check first state for collision
            s1 = path.getState(i)
            if not self.space_info.isValid(s1):
                return False

            # get bresenham line to the next state
            s2 = path.getState(i+1)
            x1 = int(s1.getX())
            y1 = int(s1.getY())
            x2 = int(s2.getX())
            y2 = int(s2.getY())
            xx, yy = skimage.draw.line(x1, y1, x2, y2)

            # check each pixel in line
            for x, y in zip(xx,yy):
                if not is_pixel_free_space(self.ppm_config_space.getPixel(y.item(), x.item())):
                    return False
        
        return True

    def cast_ray(self, state, theta, resolution):
        '''cast a ray from current state until it intersects an invalid state, return length
        This is like a simulated lidar

        Args:
            theta : float
                angle of ray relative to state yaw in radians, postive counter-clockwise
            resolution : float
                step length of ray for validity checking (i.e. obstacle collision)
        '''
        assert resolution > 0
        # initialize a state for the casted ray
        # TODO: test that changes to cloned state don't affect orignal state
        cast_state = self.space_info.cloneState(state)
        ray_heading = state.getYaw() + theta
        cast_state.setYaw(ray_heading)

        # initialize the distance of the casted ray
        ray_length = 0.0
        cast_x = state.getX()
        cast_y = state.getY()

        # propagate ray until it hits an invalid state
        # TODO: add a max range so this won't be an infinite loop
        # if isValid is somehow always True
        while self.space_info.isValid(cast_state):
            cast_x += resolution * np.cos(ray_heading)
            cast_y += resolution * np.sin(ray_heading)
            cast_state.setX(cast_x)
            cast_state.setY(cast_y)
            ray_length += resolution

        return ray_length

    def observeState(self, state):
        '''query observation from a particular state
        Args:
            state : ob.SE2State
                state from which to make observation
        Returns:
            observation : list-like
                array giving observation values
        '''
        assert self.lidar_resolution is not None
        assert self.lidar_resolution > 0
        assert self.lidar_angles is not None
        assert len(self.lidar_angles) > 0
        return [self.cast_ray(state, theta, self.lidar_resolution) for theta in self.lidar_angles]

    def state_ompl_to_numpy(self, omplState, npState=None):
        """ redirect to static method """
        return DubinsPPMSetup.static_state_ompl_to_numpy(omplState=omplState, npState=npState)

    @staticmethod
    def static_state_ompl_to_numpy(omplState, npState=None):
        """convert dubins ompl state to numpy array

        Args:
            omplState : ob.State
                ompl state object
            npState : ArrayLike OR None
                state represented in numpy array in [x, y, yaw] ordering
                if None, return np array, otherwise modify in place
        """
        ret = False
        if npState is None:
            npState = np.empty(3,)
            ret = True

        npState[0] = omplState.getX()
        npState[1] = omplState.getY()
        npState[2] = omplState.getYaw()

        if ret:
            return npState
        
    def state_numpy_to_ompl(self, npState, omplState):
        """ redirect to static method """
        return DubinsPPMSetup.static_state_numpy_to_ompl(npState=npState, omplState=omplState)

    @staticmethod
    def static_state_numpy_to_ompl(npState, omplState):
        """convert dubins state from numpy array to ompl object in-place

        Args:
            npState : ArrayLike
                state represented in numpy array in [x, y, yaw] ordering
            omplState : ob.State
                ompl state object to be modified in place
        """
        omplState.setX(npState[0])
        omplState.setY(npState[1])
        omplState.setYaw(npState[2])

    def control_ompl_to_numpy(self, omplCtrl, npCtrl=None):
        """ redirect to static method """
        return DubinsPPMSetup.static_control_ompl_to_numpy(omplCtrl=omplCtrl, npCtrl=npCtrl)

    @staticmethod
    def static_control_ompl_to_numpy(omplCtrl, npCtrl=None):
        """convert Dubins ompl control object to numpy array

        Note: this is static so that it can be called elsewhere (e.g. within StatePropagator
        class) without creating a dummy instance of the SystemSetup object.
        Note: at the same time, control_ompl_to_numpy is not made static itself because 
        we need to overload it at the parent class SystemSetup level 

        Args:
            omplCtrl : oc.Control
                dubins control in ompl RealVectorControl format
            npCtrl : ndarray (1,)
                dubins control (turnrate in [rad/sec]) represented in np array with 
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
        """redirect to static method"""
        return DubinsPPMSetup.static_control_numpy_to_ompl(omplCtrl=omplCtrl, npCtrl=npCtrl)

    @staticmethod
    def static_control_numpy_to_ompl(npCtrl, omplCtrl):
        """convert dubins control from numpy array to ompl control object in-place

        Note: this is static so that it can be called elsewhere (e.g. within StatePropagator
        class) without creating a dummy instance of the SystemSetup object.
        Note: at the same time, control_numpy_to_ompl is not made static itself because 
        we need to overload it at the parent class SystemSetup level 


        Args:
            npCtrl : ArrayLike (1,)
                Dubins control represented in np array [turnrate]
            omplCtrl : oc.Control
                Dubins control [turnrate] in ompl RealVectorControl format
        """

        assert npCtrl.shape == (1,), "Unexpected shape {}".format(npCtrl.shape)
        omplCtrl[0] = npCtrl[0]

    def dummyRiskMetric(self, state, trajectory, distance, branch_fact, depth, n_steps, policy='uniform_random', samples=None):
        '''A stand-in for the actual risk metric estimator used for model training testing purposes
        
        Args:
            state : ob.State
                state at which to evaluate risk metric
            trajectory : oc.PathControl
                trajectory arriving at state 
            distance : double
                state-space-specific distance to sample within
            branch_fact : int
                number of samples to draw
            depth : int
                number of recursive steps to estimate risk
            n_steps : int
                number of intermediate steps in sample paths
            policy : str
                string description of policy to use
            samples : list[oc.PathControl]
                list of pre-specified to samples for deterministic calc

        Returns:
            risk_est : float
                coherent risk metric estimate at state
        '''

        return state.getX()/self.space_info.getStateSpace().getBounds().high[0]