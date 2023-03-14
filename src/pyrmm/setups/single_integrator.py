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
from pyrmm.dynamics.simple_integrators import ode_1d_single_integrator as ode_1d

class SingleIntegrator1DSetup(SystemSetup):
    ''' Dubins car with ppm file for obstacle configuration space
    '''
    def __init__(self, min_speed, max_speed, lower_bound, upper_bound):
        '''
        Args:
            min_speed : float
                min linear speed of vehicle
            max_speed : float
                max linear speed of vehicle
            lower_bound : float
                lower bound of configuration space
            upper_bound : float
                upper bound of configuration space
        '''

        assert min_speed <= max_speed
        assert lower_bound <= upper_bound

        # create state space and set bounds
        state_space = ob.RealVectorStateSpace(1)
        sbounds = ob.RealVectorBounds(1)
        sbounds.setLow(0, lower_bound)
        sbounds.setHigh(0, upper_bound)
        state_space.setBounds(sbounds)

        # create control space and set bounds
        control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=1)
        cbounds = ob.RealVectorBounds(1)
        cbounds.setLow(min_speed)
        cbounds.setHigh(max_speed)
        control_space.setBounds(cbounds)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)

        # create and set state validity checker
        validityChecker = ob.StateValidityCheckerFn(partial(self.isStateValid, space_info))
        space_info.setStateValidityChecker(validityChecker)

        # call parent init to create simple setup
        super().__init__(
            space_information=space_info,
            eom_ode=lambda y, t, u: ode_1d(y, t, u)
        )

    def isStateValid(self, spaceInformation, state):
        ''' check if state is within configuration space bounds
        Args:
            spaceInformation : ob.SpaceInformationPtr
                state space information as given by SimpleSetup.getSpaceInformation
            state : ob.State
                state to check for validity
        
        Returns:
            True if state in bound and not in collision with obstacles
        '''
        return spaceInformation.satisfiesBounds(state)
    
    def state_ompl_to_numpy(self, omplState, npState=None):
        """redirect to static method"""
        return state_ompl_to_numpy(omplState=omplState, npState=npState)
    
    def state_numpy_to_ompl(self, npState, omplState):
        """redirect to static method"""
        return state_numpy_to_ompl(npState=npState, omplState=omplState)
    
    def control_ompl_to_numpy(self, omplCtrl, npCtrl=None):
        """redirect to static method"""
        return control_ompl_to_numpy(omplCtrl=omplCtrl, npCtrl=npCtrl)
    
    def control_numpy_to_ompl(self, npCtrl, omplCtrl):
        """redirect to static method"""
        return control_numpy_to_ompl(omplCtrl=omplCtrl, npCtrl=npCtrl)

def state_ompl_to_numpy(omplState, npState=None):
    """convert single integrator ompl state to numpy array

    Args:
        omplState : ob.State
            single integrator state in ompl format
        npState : ndarray (1,)
            single integrator state represented in np array [position]
            if not None, input argument is modified in place, else returned
    """
    ret = False
    if npState is None:
        npState = np.empty(1,)
        ret = True
    npState[0] = omplState[0]

    if ret:
        return npState
    
def state_numpy_to_ompl(npState, omplState):
    """convert single integrator state in numpy array in to ompl format in-place

    Args:
        npState : ndarray (1,)
            single integrator state represented in np array [position]
        omplState : ob.CompoundState
            single integrator state in ompl format
    """
    omplState[0] = npState[0]
    
def control_ompl_to_numpy(omplCtrl, npCtrl=None):
    """convert single integrator ompl control object to numpy array

    Args:
        omplCtrl : oc.Control
            single integrator control in ompl RealVectorControl format
        npCtrl : ndarray (1,)
            single integrator control represented in np array [velocity]
            if not None, input argument is modified in place, else returned
    """
    ret = False
    if npCtrl is None:
        npCtrl = np.empty(1,)
        ret = True

    npCtrl[0] = omplCtrl[0]

    if ret:
        return npCtrl
    
def control_numpy_to_ompl(npCtrl, omplCtrl):
    """convert single integrator control from numpy array to ompl control object in-place

    Args:
        npCtrl : ndarray (2,)
            single integrator control represented in np array [velocity]
        omplCtrl : oc.Control
            dubins4d control in ompl RealVectorControl format
    """

    assert npCtrl.shape == (1,), "Unexpected shape {}".format(npCtrl.shape)
    omplCtrl[0] = npCtrl[0]