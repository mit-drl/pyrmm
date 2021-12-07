'''
Create SimpleSetup for Dubins Car
'''
from __future__ import division

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

        # call parent init to create simple setup
        super().__init__(
            state_space=state_space, 
            control_space=control_space, 
            state_validity_fn=self.isStateValid, 
            state_propagator=None)

    def isStateValid(self, spaceInformation, state):
        ''' check ppm image colors for obstacle collision
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
