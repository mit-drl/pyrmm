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
    def __init__(self, ppm_file, speed, min_turn_radius):
        '''
        Args:
            ppm_file : str
                file path to ppm image used as obstacle configuration space
            speed : float
                tangential speed of dubins vehicle
            min_turn_radius : float
                minimum turning radius of the dubins vehicle
        '''

        assert speed > 0
        assert min_turn_radius > 0

        # save init args for re-creation of object
        self.ppm_file = ppm_file
        self.speed = speed
        self.min_turn_radius = min_turn_radius

        # generate configuration space from ppm file
        self.ppm_config_space = ou.PPM()
        self.ppm_config_space.loadFile(self.ppm_file)

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
        control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=1)
        dtheta = self.speed/self.min_turn_radius
        cbounds = ob.RealVectorBounds(1)
        cbounds.setLow(-dtheta)
        cbounds.setHigh(dtheta)
        control_space.setBounds(cbounds)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)

        # create and set propagator class from ODEs
        propagator = DubinsPPMStatePropagator(speed=speed, spaceInformation=space_info)
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



class DubinsPPMStatePropagator(oc.StatePropagator):

    def __init__(self, speed, spaceInformation):
        self.speed = speed
        # self.cbounds = spaceInformation.getControlSpace().getBounds()

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
        s0 = [state.getX(), state.getY(), state.getYaw()]
        t = [0.0, duration]

        # clip the control to ensure it is within the control bounds
        cbounds = self.__si.getControlSpace().getBounds()
        bounded_control = np.clip([control[0]], cbounds.low, cbounds.high)

        # call scipy's ode integrator
        sol = odeint(ode_dubins, s0, t, args=(bounded_control, self.speed))

        # store solution in result
        result.setX(sol[-1,0])
        result.setY(sol[-1,1])
        result.setYaw(sol[-1,2])

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

        # unpack objects from space information for ease of use
        cspace = self.__si.getControlSpace()
        nctrldims = cspace.getDimension()
        cbounds = cspace.getBounds()
        nsteps = path.getStateCount()
        assert nsteps >= 2
        assert nctrldims == 1
        assert duration > 0 and not np.isclose(duration, 0.0)

        # package init state and time vector
        s0 = [state.getX(), state.getY(), state.getYaw()]
        
        # create equi-spaced time vector based on number or elements
        # in path object
        t = np.linspace(0.0, duration, nsteps)

        # clip the control to ensure it is within the control bounds
        bounded_control = [np.clip(control[i], cbounds.low[i], cbounds.high[i]) for i in range(nctrldims)]
        # bounded_control = np.clip([control[0]], self.cbounds.low, self.cbounds.high)

        # call scipy's ode integrator
        sol = odeint(ode_dubins, s0, t, args=(bounded_control, self.speed))

        # store each intermediate point in the solution as pat of the path
        pstates = path.getStates()
        pcontrols = path.getControls()
        ptimes = path.getControlDurations()
        assert len(pcontrols) == len(ptimes) == nsteps-1
        for i in range(nsteps-1):
            pstates[i].setX(sol[i,0])
            pstates[i].setY(sol[i,1])
            pstates[i].setYaw(sol[i,2])
            for j in range(nctrldims):
                pcontrols[i][j] = bounded_control[j]
            ptimes[i] = t[i+1] - t[i]
        
        # store final state
        pstates[-1].setX(sol[-1,0])
        pstates[-1].setY(sol[-1,1])
        pstates[-1].setYaw(sol[-1,2])

    def canPropagateBackwards(self):
        return False

    def steer(self, from_state, to_state, control, duration):
        return False

    def canSteer(self):
        return False
