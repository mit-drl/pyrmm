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

from ompl import base as ob
from ompl import control as oc

import pybullet as pb

from pyrmm.setups import SystemSetup
from pyrmm.dynamics.quadrotor import copy_state_pb2ompl

class QuadrotorPyBulletSetup(SystemSetup):
    ''' Quadrotor vehicle defined with PyBullet physics and configuration space
    '''
    def __init__(self):
        '''
        Args:
        '''

        # TODO: assert physical inputs

        # TODO: save init args for re-creation of object

        # TODO: generate configuration space

        # create compound state space (pos, quat, vel, ang_vel) and set bounds
        state_space = ob.CompoundStateSpace()
        state_space.addSubspace(ob.RealVectorStateSpace(3), 1.0)    # position
        state_space.addSubspace(ob.SO3StateSpace(), 1.0)    # orientation
        state_space.addSubspace(ob.RealVectorStateSpace(3), 1.0)    # velocity
        state_space.addSubspace(ob.RealVectorStateSpace(3), 1.0)    # angular velocity

        # TODO: create control space and set bounds
        control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=4)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)
        # space_info = oc.SpaceInformation(stateSpace=state_space)
        self.space_info = space_info

        # TODO: create and set propagator class from ODEs

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

        # Store information about space propagator operates on
        # NOTE: this serves the same purpose asthe  protected attribute si_ 
        # but si_ does not seem to be accessible in python
        # Ref: https://ompl.kavrakilab.org/classompl_1_1control_1_1StatePropagator.html
        self.__si = spaceInformation
        super().__init__(si=spaceInformation)

    def body_thrust_torque_physics(self, controls, drone_id):
        '''physics model based on body-fixed thrust (z-axis aligned force) and torque controls
        Args:
            controls : ndarray
                (4)-shaped array of force and torques in body axes: [F_zb, M_xb, M_yb, M_zb]
            drone_id : int
                PyBullet unique object of drone

        Ref:
            Allen, "A real-time framework for kinodynamic planning in dynamic environments with application to quadrotor obstacle avoidance",
            Sec 4.1
        '''
        pb.applyExternalForce(objectUniqueId=drone_id,
                                linkIndex=-1,
                                forceObj=[0, 0, controls[0]],
                                posObj=[0, 0, 0],
                                flags=pb.LINK_FRAME,
                                physicsClientId=self.pbClientId
                                )
        pb.applyExternalTorque(objectUniqueId=drone_id,
                              linkIndex=-1,
                              torqueObj=controls[1:],
                              flags=pb.LINK_FRAME,
                              physicsClientId=self.pbClientId
                              )

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

        # TODO: reset the pb body state based on propogation start state

        # TODO: clip the control to ensure it is within the control bounds

        # call pybullet's simulator step, performs ode propagation
        # TODO: step simulation for duration
        self.body_thrust_torque_physics(control, self.pbBodyId)
        pb.stepSimulation(physicsClientId=self.pbClientId)

        # Extract state information from pb physics client and store in OMPL result
        copy_state_pb2ompl(
            pbBodyId=self.pbBodyId, 
            pbClientId=self.pbClientId, 
            omplState=result)

        
