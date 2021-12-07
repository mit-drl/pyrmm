'''
Create SimpleSetup for Dubins Car
'''
from __future__ import division
from functools import partial

from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup

class DubinsPPMSetup(SystemSetup):
    ''' Dubins car with ppm file for obstacle configuration space
    '''
    def __init__(self, ppm_file, speed, turning_radius):
        '''
        Args:
            ppm_file : str
                file path to ppm image used as obstacle configuration space
            speed : float
                tangential speed of dubins vehicle
            turning_radius : float
                turning radius of the dubins vehicle
        '''

        assert speed > 0
        assert turning_radius > 0

        # generate configuration space from ppm file
        ppm_file = ppm_file
        ppm_config_space = ou.PPM()
        ppm_config_space.loadFile(ppm_file)

        # create state space and set bounds
        state_space = ob.DubinsStateSpace(turningRadius=turning_radius)
        sbounds = ob.RealVectorBounds(2)
        sbounds.setLow(0, 0.0)
        sbounds.setHigh(0, ppm_config_space.getWidth())
        sbounds.setLow(1, 0.0)
        sbounds.setHigh(1, ppm_config_space.getHeight())
        state_space.setBounds(sbounds)

        # create control space and set bounds
        control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=1)
        dtheta = speed/turning_radius
        cbounds = ob.RealVectorBounds(1)
        cbounds.setLow(-dtheta)
        cbounds.setHigh(dtheta)
        control_space.setBounds(cbounds)

        # create state validity checker

        # call parent init to create simple setup
        super().__init__(state_space=state_space, control_space=control_space, state_validity_checker=None, state_propagator=None)
