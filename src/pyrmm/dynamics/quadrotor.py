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
    raise NotImplementedError()

    

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
        objectUniqueId=self.pbBodyId,
        physicsClientId=self.pbClientId)
    v_bu_wu__wu, w_bu_wu = pb.getBaseVelocity(
        bodyUniqueId=self.pbBodyId,
        physicsClientId=self.pbClientId
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
