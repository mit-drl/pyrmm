'''
Create SimpleSetup for Dubins Car
'''
from __future__ import division

import numpy as np

# from functools import partial
from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
from pyrmm.utils.utils import partialclass

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
        self.ppm_config_space = ou.PPM()
        self.ppm_config_space.loadFile(ppm_file)

        # create state space and set bounds
        state_space = ob.DubinsStateSpace(turningRadius=turning_radius)
        sbounds = ob.RealVectorBounds(2)
        sbounds.setLow(0, 0.0)
        sbounds.setHigh(0, self.ppm_config_space.getWidth())
        sbounds.setLow(1, 0.0)
        sbounds.setHigh(1, self.ppm_config_space.getHeight())
        state_space.setBounds(sbounds)

        # create control space and set bounds
        control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=1)
        dtheta = speed/turning_radius
        cbounds = ob.RealVectorBounds(1)
        cbounds.setLow(-dtheta)
        cbounds.setHigh(dtheta)
        control_space.setBounds(cbounds)

        # create a partially-implemented propagator class
        # NOTE: passing a class instead of a propagate func was necessary to avoid
        # lvalue conversion error: 
        # https://stackoverflow.com/questions/20825662/boost-python-argument-types-did-not-match-c-signature
        # NOTE: partially constructing the class was necessary to pass Dubins-specific "speed" attribute
        # without passing the spaceInformation that is only available in SystemSetup
        propagator_partial_cls = partialclass(DubinsPPMStatePropagator, speed)

        # call parent init to create simple setup
        super().__init__(
            state_space=state_space, 
            control_space=control_space, 
            state_validity_fn=self.isStateValid, 
            propagator_cls=propagator_partial_cls)

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
        if spaceInformation.satisfiesBounds(state):

            # if in state space bounds, check collision based
            # on ppm pixel color
            w = int(state.getX())
            h = int(state.getY())

            c = self.ppm_config_space.getPixel(h, w)
            tr = c.red > 127
            tg = c.green > 127
            tb = c.green > 127
            return tr and tg and tb

        else:
            return False

class DubinsPPMStatePropagator(oc.StatePropagator):

    def __init__(self, speed, spaceInformation):
        self.speed = speed
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
        '''
        result.setX(state.getX() + self.speed * duration * np.cos(state.getYaw()))
        result.setY(state.getY() + self.speed * duration * np.sin(state.getYaw()))
        result.setYaw(state.getYaw() + control[0] * duration)

    def canPropagateBackwards(self):
        return False

    def steer(self, from_state, to_state, control, duration):
        return False

    def canSteer(self):
        return False
