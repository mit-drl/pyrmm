'''
SystemSetup for Quadrotor vehicle

# Variable Notation:
# v__x: vector expressed in "x" frame
# q_x_y: quaternion of "x" frame with respect to "y" frame
# p_x_y__z: position of "x" frame with respect to "y" frame expressed in "z" coordinates
# v_x_y__z: velocity of "x" frame with respect to "y" frame expressed in "z" coordinates
# R_x2y: rotation matrix that maps vector represented in frame "x" to representation in frame "y" (right-multiply column vec)
#
# Frame Subscripts:
# dc = downward-facing camera (body-fixed, non-inertial frame. Origin: downward camera focal plane. Alignment with respect to drone airframe: x-forward, y-right, z-down)
# fc = forward-facing camera (body-fixed, non-inertial frame. Origin: forward camera focal plane. Alignment with respect to drone airframe: x-right, y-down, z-forward)
# bu = body-up frame (body-fixed, non-inertial frame. Origin: drone center of mass. Alignment with respect to drone airframe: x-forward, y-left, z-up)
# bd = body-down frame (body-fixed, non-inertial frame. Origin: drone center of mass. Alignment with respect to drone airframe: x-forward, y-right, z-down)
# wu = world-up frame (world-fixed, inertial frame. Origin: arbitrarily defined. Alignment with respect to world: x-(arbitrary), y-(arbitrary), z-up)
# wd = world-down frame (body-fixed, non-inertial frame. Origin: drone center of mass. Alignment with respect to world: x-(arbitrary), y-(arbitrary), z-down)
# lenu = local East-North-Up world frame (world-fixed, inertial frame. Origin: apprx at take-off point, but not guaranteed. Alignment with respect to world: x-East, y-North, z-up)
# lned = local North-East-Down world frame (world-fixed, inertial frame. Origin: apprx at take-off point, but not guaranteed. Alignment with respect to world: x-North, y-East, z-down)
# m = marker frame (inertial or non-inertial, depending on motion of marker. Origin: center of marker. Alignment when looking at marker: x-right, y-up, z-out of plane toward you)

'''

import pathlib
import numpy as np
import pybullet as pb

from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD

# hard-coded setup parameters
_QUAD_URDF = str(pathlib.Path(U.get_repo_path()).joinpath("tests/quadrotor.urdf"))
_FZMAX=8.0
_MXMAX=2.0
_MYMAX=2.0
_MZMAX=1.0

class QuadrotorPyBulletSetup(SystemSetup):
    ''' Quadrotor vehicle defined with PyBullet physics and configuration space
    '''
    def __init__(self):
        '''
        Args:
        '''

        # TODO: assert physical inputs

        # TODO: save init args for re-creation of object

        # generate configuration space
        # connect to headless physics engine
        pbClientId = pb.connect(pb.DIRECT)

        # create pybullet instance of quadrotor
        pbQuadBodyId = pb.loadURDF(_QUAD_URDF)

        # create compound state space (pos, quat, vel, ang_vel) and set bounds
        state_space = QD.QuadrotorStateSpace()

        # create control space and set bounds
        control_space = QD.QuadrotorThrustMomentControlSpace(
            stateSpace=state_space,
            fzmax=_FZMAX,
            mxmax=_MXMAX,
            mymax=_MYMAX,
            mzmax=_MZMAX)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)
        # space_info = oc.SpaceInformation(stateSpace=state_space)
        self.space_info = space_info

        # create and set propagator class from ODEs
        propagator = QuadrotorPyBulletStatePropagator(
            pbBodyId=pbQuadBodyId, 
            pbClientId=pbClientId, 
            spaceInformation=space_info)
        space_info.setStatePropagator(propagator)

        # TODO: create and set state validity checker

        # TODO: call parent init to create simple setup
        # super().__init__(space_information=space_info)

class QuadrotorPyBulletStatePropagator(oc.StatePropagator):

    def __init__(self, pbBodyId, pbClientId, spaceInformation):
        '''
        Args:
            pbBodyId :
                PyBullet unique object ID of quadrotor body associated with propagator
            pbClientId : int
                ID number of pybullet physics client
            spaceInformation : oc.SpaceInformation
                OMPL object containing information about state and control space
        '''

        self.pbClientId = pbClientId
        self.pbBodyId = pbBodyId

        # set gravity assuming world-up reference frame (positive z-axis points anti-gravity)
        pb.setGravity(0, 0, -U.GRAV_CONST, physicsClientId=self.pbClientId)

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
            This uses pybullet's built-in ode propagator.
            Ref: https://github.com/utiasDSL/gym-pybullet-drones/blob/a4e165bcbeb9133bee4bf920fca3d1a170f7bba7/gym_pybullet_drones/envs/BaseAviary.py#L272
        '''

        # reset the pb body state based on propogation start state
        QD.copy_state_ompl2pb(
            pbBodyId=self.pbBodyId, 
            pbClientId=self.pbClientId, 
            omplState=state
        )

        # clip the control to ensure it is within the control bounds
        bounded_control = U.clip_control(controlSpace=self.__si.getControlSpace(), control=control)

        # call pybullet's simulator step, performs ode propagation
        dt = pb.getPhysicsEngineParameters()['fixedTimeStep']
        t = 0.0
        while t < duration:
            # must reset external forces and torques after each sim step
            # ref: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.mq73m9o2gcpy
            QD.body_thrust_torque_physics(bounded_control, self.pbBodyId, self.pbClientId)
            pb.stepSimulation(physicsClientId=self.pbClientId)
            t += dt

        # Extract state information from pb physics client and store in OMPL result
        QD.copy_state_pb2ompl(
            pbBodyId=self.pbBodyId, 
            pbClientId=self.pbClientId, 
            omplState=result)

        
