'''
Create SimpleSetup for Dubins Car
'''
from __future__ import division

import numpy as np
from scipy.integrate import odeint

from functools import partial
from ompl import util as ou
from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
# from pyrmm.utils.utils import partialclass

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

        # generate configuration space from ppm file
        ppm_file = ppm_file
        self.ppm_config_space = ou.PPM()
        self.ppm_config_space.loadFile(ppm_file)

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
        dtheta = speed/min_turn_radius
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


def dubinsODE(y, t, u, speed):
            '''dubins vehicle ordinary differential equations
            
            Args:
                q : ???
                    state variable vector [x, y, theta]
                t : ???
                    time variable
                u : np.array
                    control vector [dtheta]
                speed : float
                    constant tangential speed
            '''

            dydt = 3*[None]
            dydt[0] = speed * np.cos(y[2])
            dydt[1] = speed * np.sin(y[2])
            dydt[2] = u[0]
            return dydt

class DubinsPPMStatePropagator(oc.StatePropagator):

    def __init__(self, speed, spaceInformation):
        self.speed = speed
        self.cbounds = spaceInformation.getControlSpace().getBounds()
        super().__init__(si=spaceInformation)

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
            # result : ob.State
            #     end state of propagation, modified in place
            # nsteps : int
            #     number of discrete steps in path

        Returns:
            None


        Notes:
            This function is similar, but disctinct from 'StatePropagator.propogate', thus its different name to no overload `propagate`. 
            propogate does not store or return the path to get to result
            By default, propagate does not perform or is used in integration,
            even when defined through an ODESolver; see:
            https://ompl.kavrakilab.org/RigidBodyPlanningWithODESolverAndControls_8py_source.html
            https://ompl.kavrakilab.org/classompl_1_1control_1_1StatePropagator.html#a4bf54becfce458e1e8abfa4a37ae8dff
            Therefore we must implement an ODE solver ourselves.
            Currently using scipy's odeint. This creates a dependency on scipy and is likely inefficient
            because it's perform the numerical integration in python instead of C++. 
            Could be improved later
        '''

        # package init state and time vector
        s0 = [state.getX(), state.getY(), state.getYaw()]
        
        # create equi-spaced time vector based on number or elements
        # in path object
        nsteps = path.getStateCount()
        assert nsteps >= 2
        t = np.linspace(0.0, duration, nsteps)

        # clip the control to ensure it is within the control bounds
        bounded_control = np.clip([control[0]], self.cbounds.low, self.cbounds.high)

        # call scipy's ode integrator
        sol = odeint(dubinsODE, s0, t, args=(bounded_control, self.speed))

        # store each intermediate point in the solution as pat of the path
        pstates = path.getStates()
        pcontrols = path.getControls()
        ptimes = path.getControlDurations()
        assert len(pcontrols) == len(ptimes) == nsteps-1
        for i in range(nsteps-1):
            pstates[i].setX(sol[i,0])
            pstates[i].setY(sol[i,1])
            pstates[i].setYaw(sol[i,2])
            pcontrols[i] = control
            ptimes[i] = t[i+1] - t[i]
        
        # store final state
        pstates[-1].setX(sol[-1,0])
        pstates[-1].setY(sol[-1,1])
        pstates[-1].setYaw(sol[-1,2])

        # store final state in result
        # result.setX(sol[-1,0])
        # result.setY(sol[-1,1])
        # result.setYaw(sol[-1,2])

        # store path to result
        # raise NotImplementedError


    def canPropagateBackwards(self):
        return False

    def steer(self, from_state, to_state, control, duration):
        return False

    def canSteer(self):
        return False
