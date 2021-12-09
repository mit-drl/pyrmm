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
        dtheta = speed/turning_radius
        cbounds = ob.RealVectorBounds(1)
        cbounds.setLow(-dtheta)
        cbounds.setHigh(dtheta)
        control_space.setBounds(cbounds)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)

        # create and set propagator class from ODEs
        # Ref: https://ompl.kavrakilab.org/RigidBodyPlanningWithODESolverAndControls_8py_source.html
        # ode = oc.ODE(dubinsODE)
        # odeSolver = oc.ODEBasicSolver(si=space_info, ode=ode)
        # propagator = oc.ODESolver.getStatePropagator(odeSolver)
        propagator = DubinsPPMStatePropagator(speed=speed, spaceInformation=space_info)
        space_info.setStatePropagator(propagator)

        # create and set state validity checker
        validityChecker = ob.StateValidityCheckerFn(partial(self.isStateValid, space_info))
        space_info.setStateValidityChecker(validityChecker)

        # create a partially-implemented propagator class
        # NOTE: passing a class instead of a propagate func was necessary to avoid
        # lvalue conversion error: 
        # https://stackoverflow.com/questions/20825662/boost-python-argument-types-did-not-match-c-signature
        # NOTE: partially constructing the class was necessary to pass Dubins-specific "speed" attribute
        # without passing the spaceInformation that is only available in SystemSetup
        # propagator_partial_cls = partialclass(DubinsPPMStatePropagator, speed)

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

# def dubinsODE(q, u, qdot):
#     '''dubins vehicle ordinary differential equations
    
#     Args:
#         q : np.array
#             state vector [x, y, theta]
#         u : np.array
#             control vector [dtheta]
#         qdot : np.array
#             state time derivative [dx, dy, dtheta], modified in place
#     '''
#     # if len(q) == 0:
#     #     return
#     theta = q[2]
#     # qdot[0] = speed * np.cos(theta)
#     # qdot[1] = speed * np.sin(theta)
#     qdot[0] = np.cos(theta)
#     qdot[1] = np.sin(theta)
#     qdot[2] = u[0]

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

        # call scipy's ode integrator
        sol = odeint(dubinsODE, s0, t, args=(control, self.speed))

        # store solution in result
        result.setX(sol[-1,0])
        result.setY(sol[-1,1])
        result.setYaw(sol[-1,2])

        # result.setX(state.getX() + self.speed * duration * np.cos(state.getYaw()))
        # result.setY(state.getY() + self.speed * duration * np.sin(state.getYaw()))
        # result.setYaw(state.getYaw() + control[0] * duration)

    def canPropagateBackwards(self):
        return False

    def steer(self, from_state, to_state, control, duration):
        return False

    def canSteer(self):
        return False
