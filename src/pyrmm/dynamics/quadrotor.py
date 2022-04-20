"""
Functions for defining and using Quadrotor dynamics

Ref: https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

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

"""

import pybullet as pb
from ompl import base as ob
from ompl import control as oc

class QuadrotorStateSpace(ob.CompoundStateSpace):
    '''6DoF state space defined as OMPL CompoundStateSpace'''
    def __init__(self):
        super().__init__()
        self.addSubspace(ob.RealVectorStateSpace(3), 1.0)   # position
        self.addSubspace(ob.SO3StateSpace(), 1.0)           # orientation
        self.addSubspace(ob.RealVectorStateSpace(3), 1.0)   # velocity
        self.addSubspace(ob.RealVectorStateSpace(3), 1.0)   # angular velocity

class QuadrotorThrustMomentControlSpace(oc.RealVectorControlSpace):
    '''Quadrotor control via body z-axis thrust and moments'''
    def __init__(self, stateSpace, fzmax, mxmax, mymax, mzmax):
        '''
        Args:
            stateSpace : ob.StateSpace
                state space associated with quadrotor (pos, orn, vel, ang vel)
            famx : float
                max net force from all rotors along body z axis (N)
            mxmax : float
                max net moment applied by rotors along body x axis (N*m)
            mymax : float
                max net moment applied by rotors along body y axis (N*m)
            mzmax : float
                max net moment applied by rotors along body z axis (N*m)
        '''
        assert fzmax > 0
        assert mxmax > 0
        assert mymax > 0
        assert mzmax > 0
        super().__init__(stateSpace=stateSpace, dim=4)

        # set control bounds: fz always positive, moments symmetric positiv and negative
        cbounds = ob.RealVectorBounds(4)
        cbounds.setLow(0, 0); cbounds.setHigh(0, fzmax)
        cbounds.setLow(1, -mxmax); cbounds.setHigh(1, mxmax)
        cbounds.setLow(2, -mymax); cbounds.setHigh(2, mymax)
        cbounds.setLow(3, -mzmax); cbounds.setHigh(3, mzmax)
        self.setBounds(cbounds)



def copy_state_ompl2pb(pbBodyId, pbClientId, omplState):
    '''Copy the 6DoF state from OMPL into PyBullet state in place
    Args:
        pbBodyId : int
            PyBullet unique object ID of quadrotor body associated with propagator
        pbClientId : int
            ID number of pybullet physics client
        omplState : ob.CompoundState
            ompl's state object for 6DoF with ordering (position, orientation, velocity, angular velocity)

    Refs:
        PyBullet quaternion ordering: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.vy9p26bpc9ft
        OMPL quaternion: https://ompl.kavrakilab.org/classompl_1_1base_1_1SO3StateSpace_1_1StateType.html
    '''
    # define position, orientation, velocity, and angular velocity lists
    p_bu_wu__wu = 3*[None]
    q_bu_wu = 4*[None]
    v_bu_wu__wu = 3*[None]
    w_bu_wu = 3*[None]

    # position
    p_bu_wu__wu[0] = omplState[0][0]
    p_bu_wu__wu[1] = omplState[0][1]
    p_bu_wu__wu[2] = omplState[0][2]

    # orientation
    q_bu_wu[0] = omplState[1].x
    q_bu_wu[1] = omplState[1].y
    q_bu_wu[2] = omplState[1].z
    q_bu_wu[3] = omplState[1].w

    # velocity
    v_bu_wu__wu[0] = omplState[2][0]
    v_bu_wu__wu[1] = omplState[2][1]
    v_bu_wu__wu[2] = omplState[2][2]

    # angular velocity
    w_bu_wu[0] = omplState[3][0]
    w_bu_wu[1] = omplState[3][1]
    w_bu_wu[2] = omplState[3][2]

    # update PyBullet State
    pb.resetBasePositionAndOrientation(
        bodyUniqueId=pbBodyId,
        posObj=p_bu_wu__wu,
        ornObj=q_bu_wu,
        physicsClientId=pbClientId
    )
    pb.resetBaseVelocity(
        objectUniqueId=pbBodyId,
        linearVelocity=v_bu_wu__wu,
        angularVelocity=w_bu_wu,
        physicsClientId=pbClientId
    )

def body_thrust_torque_physics(control, pbQuadId, pbClientId):
        '''physics model based on body-fixed thrust (z-axis aligned force) and torque controls
        Args:
            controls : oc.ControlType
                OMPL ControlType from QuadrotorThrustMomentControlSpace defining
                force and torques in body axes: [F_zb, M_xb, M_yb, M_zb]
            pbBodyId : int
                PyBullet unique object ID of quadrotor body associated with propagator
            pbClientId : int
                ID number of pybullet physics client

        Ref:
            Allen, "A real-time framework for kinodynamic planning in dynamic environments with application to quadrotor obstacle avoidance",
            Sec 4.1
        '''
        # extract and apply body z-axis aligned thrus
        thrust = control[0]
        pb.applyExternalForce(objectUniqueId=pbQuadId,
                                linkIndex=-1,
                                forceObj=[0, 0, thrust],
                                posObj=[0, 0, 0],
                                flags=pb.LINK_FRAME,
                                physicsClientId=pbClientId
                                )

        # extract and apply body torques
        moments = [control[1], control[2], control[3]]
        pb.applyExternalTorque(objectUniqueId=pbQuadId,
                              linkIndex=-1,
                              torqueObj=moments,
                              flags=pb.LINK_FRAME,
                              physicsClientId=pbClientId
                              )

def copy_state_pb2ompl(pbBodyId, pbClientId, omplState):
    '''Copy the 6DoF state from PyBullent into OMPL state in place
    Args:
        pbBodyId : int
            PyBullet unique object ID of quadrotor body associated with propagator
        pbClientId : int
            ID number of pybullet physics client
        omplState : ob.CompoundState
            ompl's state object for 6DoF with ordering (position, orientation, velocity, angular velocity)

    Refs:
        PyBullet quaternion ordering: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.vy9p26bpc9ft
        OMPL quaternion: https://ompl.kavrakilab.org/classompl_1_1base_1_1SO3StateSpace_1_1StateType.html
    '''
    p_bu_wu__wu, q_bu_wu = pb.getBasePositionAndOrientation(
        bodyUniqueId=pbBodyId,
        physicsClientId=pbClientId)
    v_bu_wu__wu, w_bu_wu = pb.getBaseVelocity(
        bodyUniqueId=pbBodyId,
        physicsClientId=pbClientId
    )

    # store state in OMPL CompoundState object in-place
    # position
    omplState[0][0] = p_bu_wu__wu[0]
    omplState[0][1] = p_bu_wu__wu[1]
    omplState[0][2] = p_bu_wu__wu[2]
    # orientation
    omplState[1].x = q_bu_wu[0]
    omplState[1].y = q_bu_wu[1]
    omplState[1].z = q_bu_wu[2]
    omplState[1].w = q_bu_wu[3]
    # velocity
    omplState[2][0] = v_bu_wu__wu[0]
    omplState[2][1] = v_bu_wu__wu[1]
    omplState[2][2] = v_bu_wu__wu[2]
    # angular velocity
    omplState[3][0] = v_bu_wu__wu[0]
    omplState[3][1] = v_bu_wu__wu[1]
    omplState[3][2] = v_bu_wu__wu[2]
