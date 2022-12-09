'''
Create SystemSetup for 1-D single integrator
'''
from __future__ import division

import numpy as np
from scipy.integrate import odeint
from functools import partial

from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
from pyrmm.dynamics.simple_integrators import ode_1d_double_integrator as ode_1d

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
        assert acc_bounds[0] < acc_bounds[1]
        assert obst_bounds[0] < obst_bounds[1]

        # save obstacle
        self.obst_bounds = obst_bounds

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
        if state[0] >= self.obst_bounds and state[0] <= self.obst_bounds[1]:
            return False
        else: 
            return True

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