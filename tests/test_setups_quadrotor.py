import pytest
import pathlib
import pickle
import numpy as np
import pybullet as pb
import pybullet_data as pbd

from copy import deepcopy
from hypothesis import given
from hypothesis import strategies as st
from ompl import control as oc

import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD
from pyrmm.setups.quadrotor import \
    QuadrotorPyBulletSetup, \
    copy_state_ompl2pb, copy_state_pb2ompl, \
    update_pickler_quadrotorstate

QUAD_URDF = str(pathlib.Path(__file__).parent.absolute().joinpath("quadrotor.urdf"))

# def get_quadrotor_pybullet_propagator_objects():
#     # connect to headless physics engine
#     pbClientId = pb.connect(pb.DIRECT)

#     # create pybullet instance of quadrotor
#     pbQuadBodyId = pb.loadURDF(QUAD_URDF)

#     # create OMPL space information object
#     sspace = QD.QuadrotorStateSpace()
#     cspace = QD.QuadrotorThrustMomentControlSpace(
#         stateSpace=sspace,
#         fzmax=8.0,
#         mxmax=2.0,
#         mymax=2.0,
#         mzmax=1.0)
#     space_info = oc.SpaceInformation(stateSpace=sspace, controlSpace=cspace)

#     # create state propagator
#     quadPropagator = QuadrotorPyBulletStatePropagator(
#         pbBodyId=pbQuadBodyId, 
#         pbClientId=pbClientId, 
#         spaceInformation=space_info)

#     return space_info, quadPropagator

# @pytest.fixture
# def quadrotor_pybullet_propagator():
#     # ~~ ARRANGE ~~

#     space_info, quadPropagator = get_quadrotor_pybullet_propagator_objects()

#     yield space_info, quadPropagator

#     # ~~ TEARDOWN ~~
#     # disconnect from pybullet physics client
#     pb.disconnect()

@pytest.fixture
def quadrotor_pybullet_setup_no_lidar():
    # ~~ ARRANGE ~~
    qpbsetup = QuadrotorPyBulletSetup()

    # ~~ PASS TO TEST
    yield qpbsetup

    # ~~ Teardown ~~
    # disconnect from pybullet physics client
    pb.disconnect()

def get_pybullet_copy_state_objects():
    # connect to headless physics engine
    pbClientId = pb.connect(pb.DIRECT)

    # create pybullet instance of quadrotor
    pbQuadBodyId = pb.loadURDF(QUAD_URDF)

    # create ompl state space and state
    omplStateSpace = QD.QuadrotorStateSpace()
    omplState = omplStateSpace.allocState()

    return pbClientId, pbQuadBodyId, omplState

@pytest.fixture
def pybullet_copy_state():
    '''arrange the pybullet and ompl objects for copy state tests, and cleanup'''

    # ~~ ARRANGE ~~

    pbClientId, pbQuadBodyId, omplState = get_pybullet_copy_state_objects()

    yield pbClientId, pbQuadBodyId, omplState

    # disconnect from pybullet physics client
    pb.disconnect()


def test_copy_state_0(pybullet_copy_state):
    '''test that a state is not modified after a pass between ompl and pb'''

    # ~~ ARRANGE ~~
    pbClientId, pbQuadBodyId, omplState = pybullet_copy_state

    # store pb state for later comparison
    old_pos, old_quat = deepcopy(pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId))
    old_vel, old_angv = deepcopy(pb.getBaseVelocity(pbQuadBodyId, pbClientId))

    # ~~ ACT ~~
    # copy pb state to ompl
    copy_state_pb2ompl(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # copy ompl state back to pb
    copy_state_ompl2pb(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # ~~ ASSERT ~~
    pos, quat = pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId)
    vel, angv = pb.getBaseVelocity(pbQuadBodyId, pbClientId)
    assert np.allclose(pos, old_pos)
    assert np.allclose(quat, old_quat)
    assert np.allclose(vel, old_vel)
    assert np.allclose(angv, old_angv)


def test_copy_state_1(pybullet_copy_state):
    '''test that a state that is modified is not equal to original state'''

    # ~~ ARRANGE ~~
    pbClientId, pbQuadBodyId, omplState = pybullet_copy_state

    # store pb state for later comparison
    old_pos, old_quat = deepcopy(pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId))
    old_vel, old_angv = deepcopy(pb.getBaseVelocity(pbQuadBodyId, pbClientId))

    # ~~ ACT ~~
    # copy pb state to ompl
    copy_state_pb2ompl(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # modify the quaternion
    omplState[1].setAxisAngle(1,0,0,np.pi/6)

    # copy ompl state back to pb
    copy_state_ompl2pb(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # ~~ ASSERT ~~
    pos, quat = pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId)
    vel, angv = pb.getBaseVelocity(pbQuadBodyId, pbClientId)
    assert np.allclose(pos, old_pos)
    assert not np.allclose(quat, old_quat)
    assert np.allclose(vel, old_vel)
    assert np.allclose(angv, old_angv)

def test_copy_state_2(pybullet_copy_state):
    '''test that a random, yet pre-specified, state is not modified between copies'''

    # ~~ ARRANGE ~~
    if pybullet_copy_state is None:
        pbClientId, pbQuadBodyId, omplState = get_pybullet_copy_state_objects()
    else:
        pbClientId, pbQuadBodyId, omplState = pybullet_copy_state

    # set pre-specified random state
    pos0 = [-3.83344849, -3.5406428 ,  2.06718344]
    eul0 = [6.24745327, 6.26306319, 0.59214794]
    quat0 = pb.getQuaternionFromEuler(eul0)
    vel0 = [3.43719412, 2.46519465, 1.83673987]
    angv0 = [ 0.80282335,  0.97132736, -0.50887632]
    pb.resetBasePositionAndOrientation(pbQuadBodyId, pos0, quat0, pbClientId)
    pb.resetBaseVelocity(pbQuadBodyId, vel0, angv0, pbClientId)

    # ~~ ACT ~~
    # copy pb state to ompl
    copy_state_pb2ompl(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # copy ompl state back to pb
    copy_state_ompl2pb(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # ~~ ASSERT ~~
    pos, quat = pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId)
    vel, angv = pb.getBaseVelocity(pbQuadBodyId, pbClientId)
    assert np.allclose(pos, pos0)
    assert np.allclose(quat, quat0)
    assert np.allclose(vel, vel0)
    assert np.allclose(angv, angv0)

def test_update_pickler_quadrotorstate_0(pybullet_copy_state):
    '''test QuadrotorStateSpace state can be pickled and reproduced'''

    # ~~ ARRANGE ~~
    if pybullet_copy_state is None:
        pbClientId, pbQuadBodyId, omplState = get_pybullet_copy_state_objects()
    else:
        pbClientId, pbQuadBodyId, omplState = pybullet_copy_state

    # set pre-specified random state
    pos0 = [-3.83344849, -3.5406428 ,  2.06718344]
    eul0 = [6.24745327, 6.26306319, 0.59214794]
    quat0 = pb.getQuaternionFromEuler(eul0)
    vel0 = [3.43719412, 2.46519465, 1.83673987]
    angv0 = [ 0.80282335,  0.97132736, -0.50887632]
    pb.resetBasePositionAndOrientation(pbQuadBodyId, pos0, quat0, pbClientId)
    pb.resetBaseVelocity(pbQuadBodyId, vel0, angv0, pbClientId)

    # copy pb state to ompl
    copy_state_pb2ompl(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # ~~ ACT ~~
    # update pickler
    update_pickler_quadrotorstate()

    # pickle and unpickle ompl state
    omplState_copy = pickle.loads(pickle.dumps(omplState))

    # ~~ ASSERT ~~
    assert np.allclose(pos0, [omplState_copy[0][i] for i in range(3)])
    qc = omplState_copy[1]
    assert np.allclose(quat0, [qc.x, qc.y, qc.z, qc.w])
    assert np.allclose(vel0, [omplState_copy[2][i] for i in range(3)])
    assert np.allclose(angv0, [omplState_copy[3][i] for i in range(3)])


def test_QuadrotorPyBulletSetup_state_checker_0(quadrotor_pybullet_setup_no_lidar):
    '''test that simple, initial state in empty environment has no collisions'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # get initial state from pybullet
    s0 = qpbsetup.space_info.allocState()
    copy_state_pb2ompl(
        pbBodyId=qpbsetup.pbBodyId,
        pbClientId=qpbsetup.pbClientId,
        omplState=s0
    )

    # ~~ ACT ~~
    # get state validity
    is_valid = qpbsetup.space_info.isValid(s0)

    # ~~ ASSERT ~~
    assert is_valid

def test_QuadrotorPyBulletSetup_state_checker_1(quadrotor_pybullet_setup_no_lidar):
    '''test that collision with floor plane causes invalid state'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # load in floor plane to environment (from pybullet_data)
    pb.setAdditionalSearchPath(pbd.getDataPath())
    floorBodyId = pb.loadURDF("plane100.urdf")

    # get initial state from pybullet
    s0 = qpbsetup.space_info.allocState()
    copy_state_pb2ompl(
        pbBodyId=qpbsetup.pbBodyId,
        pbClientId=qpbsetup.pbClientId,
        omplState=s0
    )

    # ~~ ACT ~~
    # get state validity
    is_valid = qpbsetup.space_info.isValid(s0)

    # ~~ ASSERT ~~
    assert not is_valid

def test_QuadrotorPyBulletSetup_observeLidar_0(quadrotor_pybullet_setup_no_lidar):
    '''check that ray cast to ground is of expected length'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # load in floor plane to environment (from pybullet_data)
    pb.setAdditionalSearchPath(pbd.getDataPath())
    floorBodyId = pb.loadURDF("plane100.urdf")

    # Set quadrotor pybullet state to known position
    xpos = 13.4078
    ypos = -36.29397
    zpos = 10.0
    initPos, initOrn = pb.getBasePositionAndOrientation(qpbsetup.pbBodyId)
    pb.resetBasePositionAndOrientation(
        bodyUniqueId = qpbsetup.pbBodyId,
        posObj = [xpos, ypos, zpos],
        ornObj = initOrn,
        physicsClientId = qpbsetup.pbClientId
    )
    s0 = qpbsetup.space_info.allocState()
    copy_state_pb2ompl(
        pbBodyId=qpbsetup.pbBodyId,
        pbClientId=qpbsetup.pbClientId,
        omplState=s0
    )

    # ~~ ACT ~~
    # observe lidar ray casts
    ray = qpbsetup.observeLidar(
        state = s0,
        ray_range = 100,
        ray_angles=[(np.pi, 0.0)])[0]

    # ~~ ASSERT ~~

    # check that body hit is the floor
    assert ray[0] == floorBodyId
    assert ray[1] == -1

    # check that hit fraction is 1/10 of ray length  
    # quadrotor is 10 m above ground and ray is 100 m long
    assert np.isclose(ray[2], 0.1, rtol=1e-4)

    # check world position of hit
    assert np.isclose(ray[3][0], xpos)
    assert np.isclose(ray[3][1], ypos)
    assert np.isclose(ray[3][2], 0.0, atol=1e-3)

# @given(
#     x = st.floats(min_value=-10, max_value=10, allow_nan=False),
#     y = st.floats(min_value=-10, max_value=10, allow_nan=False),
# )
def test_QuadrotorPyBulletSetup_observeLidar_1(quadrotor_pybullet_setup_no_lidar):
    '''check that batch of rays cast to ground is of expected length, others don't intersect'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # load in floor plane to environment (from pybullet_data)
    pb.setAdditionalSearchPath(pbd.getDataPath())
    floorBodyId = pb.loadURDF("plane100.urdf")

    # create OMPL state to ray cast from
    x = -5.03267
    y = 6.637698
    z = 1.0
    s0 = qpbsetup.space_info.allocState()
    copy_state_pb2ompl(
        pbBodyId=qpbsetup.pbBodyId,
        pbClientId=qpbsetup.pbClientId,
        omplState=s0
    )
    s0[0][0] = x
    s0[0][1] = y
    s0[0][2] = z

    # create lidar ray angles
    ray_range = 100
    angles = [
        (np.pi, 0),
        (3*np.pi/4, 0),
        (3*np.pi/4, np.pi/4),
        (3*np.pi/4, np.pi/2),
        (3*np.pi/4, 3*np.pi/4),
        (3*np.pi/4, np.pi),
        (3*np.pi/4, 5*np.pi/4),
        (3*np.pi/4, 3*np.pi/2),
        (3*np.pi/4, 7*np.pi/4),
        (3*np.pi/4, 2*np.pi),
        (np.pi/2, 0),
        (np.pi/2, np.pi/2),
        (np.pi/2, np.pi),
        (np.pi/2, 3*np.pi/2),
        (0,0) 
    ]
    n_rays = len(angles)
    
    # ~~ ACT ~~
    # observe lidar ray casts
    rays = qpbsetup.observeLidar(s0, ray_range=ray_range, ray_angles=angles)

    # ~~ ASSERT ~~

    # check rays that should intersect floor
    # check that body hit is the floor
    assert rays[0][0] == floorBodyId
    assert rays[0][1] == -1

    # check that hit fraction is 1/10 of ray length  
    # quadrotor is 10 m above ground and ray is 100 m long
    assert np.isclose(rays[0][2], 0.01, rtol=1e-3)

    # check world position of hit
    assert np.isclose(rays[0][3][0], x)
    assert np.isclose(rays[0][3][1], y)
    assert np.isclose(rays[0][3][2], 0.0, atol=1e-3)

    # check rays that intersect ground
    for i in range(1,10):
        ray = rays[i]
        ang = angles[i]
        assert ray[0] == floorBodyId
        assert ray[1] == -1
        assert np.isclose(ray[2], z/(np.cos(np.pi-ang[0]) * ray_range), rtol=1e-3)
        assert np.isclose(ray[3][0], x + z * np.tan(np.pi-ang[0]) * np.cos(ang[1]), rtol=1e-3)
        assert np.isclose(ray[3][1], y + z * np.tan(np.pi-ang[0]) * np.sin(ang[1]), rtol=1e-3)
        assert np.isclose(ray[3][2], 0.0, atol=1e-3)

    # check rays that don't intersect ground
    for i in range(10,n_rays):
        ray = rays[i]
        ang = angles[i]
        assert ray[0] == -1
        assert ray[1] == -1
        assert np.isclose(ray[2], 1.0)

def test_QuadrotorPyBulletSetup_observeLidar_2(quadrotor_pybullet_setup_no_lidar):
    '''check ray casts intersect with ground given arbitrary pos and orientation'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # load in floor plane to environment (from pybullet_data)
    pb.setAdditionalSearchPath(pbd.getDataPath())
    floorBodyId = pb.loadURDF("plane100.urdf")

    # ranomized but fixed position and orientation
    px = -7.502776
    py = -0.88515
    pz = 4.2003
    p_bu_wu__wu = (px, py, pz)
    roll = 0.9385
    pitch = -1.4110
    yaw = -0.0880
    q_bu_wu = pb.getQuaternionFromEuler(eulerAngles=(roll, pitch, yaw))
    
    # compute rotation matrix from body to world
    R_bu2wu = np.array(pb.getMatrixFromQuaternion(q_bu_wu)).reshape(3,3)

    # set pybullet and ompl state
    pb.resetBasePositionAndOrientation(qpbsetup.pbBodyId, p_bu_wu__wu, q_bu_wu, qpbsetup.pbClientId)
    s0 = qpbsetup.space_info.allocState()
    copy_state_pb2ompl(
        pbBodyId=qpbsetup.pbBodyId,
        pbClientId=qpbsetup.pbClientId,
        omplState=s0
    )

    # create lidar ray angles
    ray_range = 100
    ray_angles = [
        (np.pi/2, 0),
        (np.pi/2, np.pi/2),
        (np.pi/2, np.pi),
        (np.pi/2, 3*np.pi/2),
        (0,0),
        (np.pi,0)
    ]
    n_rays = len(ray_angles)

    # unit vectors of rays in body coords
    ray_unit_vectors__bu = [
        (1,0,0),
        (0,1,0),
        (-1,0,0),
        (0,-1,0),
        (0,0,1),
        (0,0,-1)
    ]

    # ~~ ACT ~~
    # observe lidar ray casts
    rays = qpbsetup.observeLidar(s0, ray_range=ray_range, ray_angles=ray_angles)

    # ~~ ASSERT ~~
    # based on randomized orientation, rays that are expected to intersect ground
    intersect_rays = [2,3,5]
    nonintersect_rays = [0,1,4]

    for i in intersect_rays:
        assert rays[i][0] == floorBodyId

        # unit vector of ray in world-up coords
        ru__wu = np.matmul(R_bu2wu, np.array(ray_unit_vectors__bu[i]).reshape(3,1))
        
        # compute expected length to intersect
        l_exp = abs(pz)/np.dot([0,0,-1], ru__wu)[0]

        # check ray length fraction
        assert np.isclose(rays[i][2], l_exp/ray_range, rtol=1e-3)


    # check rays not expected to intersect ground
    for i in nonintersect_rays:
        assert rays[i][0] == -1
        assert np.isclose(rays[i][2], 1.0)

def test_QuadrotorPyBulletSetup_observeState_0():
    '''check that simple state is observed as expected'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    ldr_angs = [(np.pi,0)]
    qpbsetup = QuadrotorPyBulletSetup(lidar_range=100.,lidar_angles=ldr_angs)

    # load in floor plane to environment (from pybullet_data)
    pb.setAdditionalSearchPath(pbd.getDataPath())
    floorBodyId = pb.loadURDF("plane100.urdf")

    # Set quadrotor pybullet state t be 10 meters off the floor
    xpos = 0
    ypos = 0
    zpos = 10.0
    initPos, initOrn = pb.getBasePositionAndOrientation(qpbsetup.pbBodyId)
    pb.resetBasePositionAndOrientation(
        bodyUniqueId = qpbsetup.pbBodyId,
        posObj = [xpos, ypos, zpos],
        ornObj = initOrn,
        physicsClientId = qpbsetup.pbClientId
    )
    s0 = qpbsetup.space_info.allocState()
    copy_state_pb2ompl(
        pbBodyId=qpbsetup.pbBodyId,
        pbClientId=qpbsetup.pbClientId,
        omplState=s0
    )

    # ~~ ACT ~~
    # observe state
    obs = qpbsetup.observeState(state = s0)

    # ~~ ASSERT ~~

    # check lidar observation
    assert np.isclose(obs[0], zpos, rtol=1e-3)

    # check orientation observation
    assert np.isclose(obs[1], s0[1].x)
    assert np.isclose(obs[2], s0[1].y)
    assert np.isclose(obs[3], s0[1].z)
    assert np.isclose(obs[4], s0[1].w)

    # check velocity observation
    assert np.isclose(obs[5], s0[2][0])
    assert np.isclose(obs[6], s0[2][1])
    assert np.isclose(obs[7], s0[2][2])

    # check angular rate observation
    assert np.isclose(obs[8], s0[3][0])
    assert np.isclose(obs[9], s0[3][1])
    assert np.isclose(obs[10], s0[3][2])

    # ~~ Teardown ~~
    # disconnect from pybullet physics client
    pb.disconnect()

def test_QuadrotorPyBulletSetup_observeState_1():
    '''check that simple state is observed as expected'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    ldr_angs = [
        (np.pi/2, 0),
        (np.pi/2, np.pi/2),
        (np.pi/2, np.pi),
        (np.pi/2, 3*np.pi/2),
        (0,0),
        (np.pi,0)
    ]
    n_rays = len(ldr_angs)
    ldr_range = 100.
    qpbsetup = QuadrotorPyBulletSetup(lidar_range=ldr_range,lidar_angles=ldr_angs)

    # unit vectors of rays in body coords
    ray_unit_vectors__bu = [
        (1,0,0),
        (0,1,0),
        (-1,0,0),
        (0,-1,0),
        (0,0,1),
        (0,0,-1)
    ]

    # load in floor plane to environment (from pybullet_data)
    pb.setAdditionalSearchPath(pbd.getDataPath())
    floorBodyId = pb.loadURDF("plane100.urdf")

    # Set quadrotor pybullet state to randomized yet fixed position and orientation
    p_bu_wu__wu = [-4.46113278,  3.40651189,  8.04537052]
    eul_bu_wu = [-0.00525912,  0.23032606,  0.7182055 ]  # roll, pitch, yaw
    q_bu_wu = pb.getQuaternionFromEuler(eulerAngles=eul_bu_wu)

    # based on randomized orientation, rays that are expected to intersect ground
    intersect_rays = [0,1,5]
    nonintersect_rays = [2,3,4]

    # compute rotation matrix from body to world
    R_bu2wu = np.array(pb.getMatrixFromQuaternion(q_bu_wu)).reshape(3,3)

    # set quadrotor state to randomized but fixed velocity and angular rates
    v_bu_wu__wu = [-8.34776217,  5.75793221,  2.64415451]
    w_bu_wu = [-0.86476772,  0.1442532 ,  1.27495011]

    # set pybullet and ompl state
    pb.resetBasePositionAndOrientation(qpbsetup.pbBodyId, p_bu_wu__wu, q_bu_wu, qpbsetup.pbClientId)
    pb.resetBaseVelocity(qpbsetup.pbBodyId, v_bu_wu__wu, w_bu_wu, qpbsetup.pbClientId)
    s0 = qpbsetup.space_info.allocState()
    copy_state_pb2ompl(
        pbBodyId=qpbsetup.pbBodyId,
        pbClientId=qpbsetup.pbClientId,
        omplState=s0
    )

    # ~~ ACT ~~
    # observe state
    obs = qpbsetup.observeState(state = s0)

    # ~~ ASSERT ~~

    # check intersecting rays
    for i in intersect_rays:

        # unit vector of ray in world-up coords
        ru__wu = np.matmul(R_bu2wu, np.array(ray_unit_vectors__bu[i]).reshape(3,1))
        
        # compute expected length to intersect
        l_exp = min(abs(p_bu_wu__wu[2])/np.dot([0,0,-1], ru__wu)[0], ldr_range)

        # check ray length fraction
        assert np.isclose(obs[i], l_exp, rtol=1e-3)

    # check rays not expected to intersect ground
    for i in nonintersect_rays:
        assert np.isclose(obs[i], ldr_range)

    # check orientation observation
    assert np.isclose(obs[n_rays], s0[1].x)
    assert np.isclose(obs[n_rays+1], s0[1].y)
    assert np.isclose(obs[n_rays+2], s0[1].z)
    assert np.isclose(obs[n_rays+3], s0[1].w)

    # check velocity observation
    assert np.isclose(obs[n_rays+4], s0[2][0])
    assert np.isclose(obs[n_rays+5], s0[2][1])
    assert np.isclose(obs[n_rays+6], s0[2][2])

    # check angular rate observation
    assert np.isclose(obs[n_rays+7], s0[3][0])
    assert np.isclose(obs[n_rays+8], s0[3][1])
    assert np.isclose(obs[n_rays+9], s0[3][2])

    # ~~ Teardown ~~
    # disconnect from pybullet physics client
    pb.disconnect()

def test_QuadrotorPyBulletStatePropagator_propagate_hover(quadrotor_pybullet_setup_no_lidar):
    '''Test that perfect hover thrust does not move the quadrotor'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # create initial and resulting OMPL state objects
    init_state = qpbsetup.space_info.allocState()
    copy_state_pb2ompl(qpbsetup.pbBodyId, qpbsetup.pbClientId, init_state)
    result_state = qpbsetup.space_info.allocState()

    # calculate hovering thrust
    mass = 0.5  # see quadrotor.urdf
    thrust = mass * U.GRAV_CONST
    control = qpbsetup.space_info.getControlSpace().allocControl()
    control[0] = thrust
    control[1] = 0.0
    control[2] = 0.0
    control[3] = 0.0

    # ~~ ACT ~~

    # apply hovering thrust
    qpbsetup.propagate(
        state=init_state,
        control=control,
        duration=10.0,
        result=result_state
    )


    # ~~ ASSERT ~~

    # check that init state is almost equal to resulting state because perfect hover
    assert np.isclose(init_state[0][0], result_state[0][0])
    assert np.isclose(init_state[0][1], result_state[0][1])
    assert np.isclose(init_state[0][2], result_state[0][2])
    assert np.isclose(init_state[1].x, result_state[1].x)
    assert np.isclose(init_state[1].y, result_state[1].y)
    assert np.isclose(init_state[1].z, result_state[1].z)
    assert np.isclose(init_state[1].w, result_state[1].w)
    assert np.isclose(init_state[2][0], result_state[2][0])
    assert np.isclose(init_state[2][1], result_state[2][1])
    assert np.isclose(init_state[2][2], result_state[2][2])
    assert np.isclose(init_state[3][0], result_state[3][0])
    assert np.isclose(init_state[3][1], result_state[3][1])
    assert np.isclose(init_state[3][2], result_state[3][2])

def test_QuadrotorPyBulletStatePropagator_propagate_drift(quadrotor_pybullet_setup_no_lidar):
    '''Test that non-zero initial horizontal velocity drifts a known distance'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # set non-zero initial horizontal velocity
    xvel =  9.4810
    dur = 6.0445
    pb.resetBaseVelocity(
        objectUniqueId=qpbsetup.pbBodyId,
        linearVelocity=[xvel, 0, 0],
        angularVelocity=[0.0, 0.0, 0.0],
        physicsClientId=qpbsetup.pbClientId)

    # turn off linear damping for perfect drift
    pb.changeDynamics(
        bodyUniqueId=qpbsetup.pbBodyId,
        linkIndex=-1,
        linearDamping=0.0
    )

    # create initial and resulting OMPL state objects
    init_state = qpbsetup.space_info.getStateSpace().allocState()
    copy_state_pb2ompl(qpbsetup.pbBodyId, qpbsetup.pbClientId, init_state)
    result_state = qpbsetup.space_info.getStateSpace().allocState()

    # calculate hovering thrust
    mass = 0.5  # see quadrotor.urdf
    thrust = mass * U.GRAV_CONST
    control = qpbsetup.space_info.getControlSpace().allocControl()
    control[0] = thrust
    control[1] = 0.0
    control[2] = 0.0
    control[3] = 0.0

    # ~~ ACT ~~

    # apply hovering thrust
    qpbsetup.propagate(
        state=init_state,
        control=control,
        duration=dur,
        result=result_state
    )

    # ~~ ASSERT ~~

    # check that init state is almost equal to resulting state because perfect hover
    assert np.isclose(result_state[0][0], xvel*dur, rtol=1e-2)
    assert np.isclose(init_state[0][1], result_state[0][1])
    assert np.isclose(init_state[0][2], result_state[0][2])
    assert np.isclose(init_state[1].x, result_state[1].x)
    assert np.isclose(init_state[1].y, result_state[1].y)
    assert np.isclose(init_state[1].z, result_state[1].z)
    assert np.isclose(init_state[1].w, result_state[1].w)
    assert np.isclose(init_state[2][0], result_state[2][0])
    assert np.isclose(result_state[2][0], xvel)
    assert np.isclose(init_state[2][1], result_state[2][1])
    assert np.isclose(init_state[2][2], result_state[2][2])
    assert np.isclose(init_state[3][0], result_state[3][0])
    assert np.isclose(init_state[3][1], result_state[3][1])
    assert np.isclose(init_state[3][2], result_state[3][2])


def test_QuadrotorPyBulletStatePropagator_propagate_climb(quadrotor_pybullet_setup_no_lidar):
    '''Test that known climb rate results in expect climb distance'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # create initial and resulting OMPL state objects
    init_state = qpbsetup.space_info.getStateSpace().allocState()
    copy_state_pb2ompl(qpbsetup.pbBodyId, qpbsetup.pbClientId, init_state)
    result_state = qpbsetup.space_info.getStateSpace().allocState()

    # turn off linear damping for no velocity damping
    pb.changeDynamics(
        bodyUniqueId=qpbsetup.pbBodyId,
        linkIndex=-1,
        linearDamping=0.0
    )

    # calculate climbing thrust
    climb_acc = -0.122648
    dur = 4.18569
    mass = 0.5  # see quadrotor.urdf
    thrust = mass * (U.GRAV_CONST + climb_acc)
    control = qpbsetup.space_info.getControlSpace().allocControl()
    control[0] = thrust
    control[1] = 0.0
    control[2] = 0.0
    control[3] = 0.0

    # ~~ ACT ~~

    # apply hovering thrust
    qpbsetup.propagate(
        state=init_state,
        control=control,
        duration=dur,
        result=result_state
    )

    # ~~ ASSERT ~~

    # check that init state is almost equal to resulting state because perfect hover
    exp_p_z = 0.5 * climb_acc * dur**2
    exp_v_z = climb_acc * dur
    assert np.isclose(init_state[0][0], result_state[0][0])
    assert np.isclose(init_state[0][1], result_state[0][1])
    assert np.isclose(result_state[0][2], exp_p_z, rtol=1e-2)
    assert np.isclose(init_state[1].x, result_state[1].x)
    assert np.isclose(init_state[1].y, result_state[1].y)
    assert np.isclose(init_state[1].z, result_state[1].z)
    assert np.isclose(init_state[1].w, result_state[1].w)
    assert np.isclose(init_state[2][0], result_state[2][0])
    assert np.isclose(init_state[2][1], result_state[2][1])
    assert np.isclose(result_state[2][2], exp_v_z, rtol=1e-2)
    assert np.isclose(init_state[3][0], result_state[3][0])
    assert np.isclose(init_state[3][1], result_state[3][1])
    assert np.isclose(init_state[3][2], result_state[3][2])


def test_QuadrotorPyBulletStatePropagator_propagate_path_hover(quadrotor_pybullet_setup_no_lidar):
    '''Test that perfect hover thrust does not move the quadrotor using propagate_path'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # create initial and resulting OMPL state objects
    init_state = qpbsetup.space_info.getStateSpace().allocState()
    copy_state_pb2ompl(qpbsetup.pbBodyId, qpbsetup.pbClientId, init_state)

    # calculate hovering thrust
    mass = 0.5  # see quadrotor.urdf
    thrust = mass * U.GRAV_CONST
    control = qpbsetup.space_info.getControlSpace().allocControl()
    control[0] = thrust
    control[1] = 0.0
    control[2] = 0.0
    control[3] = 0.0

    # create path object and alloc a randomized number of intermediate steps
    path = oc.PathControl(qpbsetup.space_info)
    nsteps = np.random.randint(5, 10)
    for _ in range(nsteps-1):
        path.append(state=qpbsetup.space_info.allocState(), control=qpbsetup.space_info.allocControl(), duration=0)
    path.append(state=qpbsetup.space_info.allocState())

    # ~~ ACT ~~

    # apply hovering thrust
    qpbsetup.propagate_path(
        state = init_state,
        control = control,
        duration = 10.0,
        path = path
    )

    # ~~ ASSERT ~~

    # check that init state is almost equal to every state in path because hovering
    for i in range(nsteps):
        result_state = path.getState(i)
        assert np.isclose(init_state[0][0], result_state[0][0])
        assert np.isclose(init_state[0][1], result_state[0][1])
        assert np.isclose(init_state[0][2], result_state[0][2])
        assert np.isclose(init_state[1].x, result_state[1].x)
        assert np.isclose(init_state[1].y, result_state[1].y)
        assert np.isclose(init_state[1].z, result_state[1].z)
        assert np.isclose(init_state[1].w, result_state[1].w)
        assert np.isclose(init_state[2][0], result_state[2][0])
        assert np.isclose(init_state[2][1], result_state[2][1])
        assert np.isclose(init_state[2][2], result_state[2][2])
        assert np.isclose(init_state[3][0], result_state[3][0])
        assert np.isclose(init_state[3][1], result_state[3][1])
        assert np.isclose(init_state[3][2], result_state[3][2])

def test_QuadrotorPyBulletStatePropagator_propagate_path_climb(quadrotor_pybullet_setup_no_lidar):
    '''Test that known climb rate results in expected climb distance at each point on path'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # create initial and resulting OMPL state objects
    init_state = qpbsetup.space_info.getStateSpace().allocState()
    copy_state_pb2ompl(qpbsetup.pbBodyId, qpbsetup.pbClientId, init_state)

    # turn off linear damping for no velocity damping
    pb.changeDynamics(
        bodyUniqueId=qpbsetup.pbBodyId,
        linkIndex=-1,
        linearDamping=0.0
    )

    # calculate climbing thrust
    climb_acc = np.random.rand()*10 - 5
    req_dur = np.random.rand()*10 + 1
    mass = 0.5  # see quadrotor.urdf
    thrust = mass * (U.GRAV_CONST + climb_acc)
    control = qpbsetup.space_info.getControlSpace().allocControl()
    control[0] = thrust
    control[1] = 0.0
    control[2] = 0.0
    control[3] = 0.0

    # create path object and alloc a randomized number of intermediate steps
    path = oc.PathControl(qpbsetup.space_info)
    nsteps = np.random.randint(5, 10)
    for _ in range(nsteps-1):
        path.append(state=qpbsetup.space_info.allocState(), control=qpbsetup.space_info.allocControl(), duration=0)
    path.append(state=qpbsetup.space_info.allocState())

    # ~~ ACT ~~

    # apply hovering thrust
    true_duration = qpbsetup.propagate_path(
        state=init_state,
        control=control,
        duration=req_dur,
        path=path,
        ret_true_duration = True
    )

    # ~~ ASSERT ~~

    # check that init state is almost equal to resulting state because perfect hover
    # subdur = dur/nsteps
    cum_t = 0.0
    for i in range(nsteps):
        exp_p_z = 0.5 * climb_acc * cum_t**2
        exp_v_z = climb_acc * cum_t
        result_state = path.getState(i)
        assert np.isclose(init_state[0][0], result_state[0][0])
        assert np.isclose(init_state[0][1], result_state[0][1])
        assert np.isclose(result_state[0][2], exp_p_z, rtol=5e-2)
        assert np.isclose(init_state[1].x, result_state[1].x)
        assert np.isclose(init_state[1].y, result_state[1].y)
        assert np.isclose(init_state[1].z, result_state[1].z)
        assert np.isclose(init_state[1].w, result_state[1].w)
        assert np.isclose(init_state[2][0], result_state[2][0])
        assert np.isclose(init_state[2][1], result_state[2][1])
        assert np.isclose(result_state[2][2], exp_v_z, rtol=1e-2)
        assert np.isclose(init_state[3][0], result_state[3][0])
        assert np.isclose(init_state[3][1], result_state[3][1])
        assert np.isclose(init_state[3][2], result_state[3][2])
        if i < nsteps-1:
            cum_t += path.getControlDurations()[i]

    assert np.isclose(true_duration, cum_t)

def test_QuadrotorPyBulletSetup_state_sampler_0(quadrotor_pybullet_setup_no_lidar):
    '''test that state sampler produces non-identical states'''

    # ~~ ARRANGE ~~
    # get quadrotor setup object
    if quadrotor_pybullet_setup_no_lidar is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup_no_lidar

    # allocate state sampler and state storage
    n_samples = 100
    sampler = qpbsetup.space_info.allocStateSampler()
    states = n_samples * [None] 

    # ~~ ACT ~~
    # get state validity
    for i in range(n_samples):
        states[i] = qpbsetup.space_info.allocState()
        sampler.sampleUniform(states[i])

    # ~~ ASSERT ~~

    for i in range(n_samples):
        pos_i = np.array([states[i][0][k] for k in range(3)])
        qut_i = np.array([states[i][1].x, states[i][1].y, states[i][1].z, states[i][1].w])
        vel_i = np.array([states[i][2][k] for k in range(3)])
        omg_i = np.array([states[i][3][k] for k in range(3)])


        # check that quaternions are valid
        assert np.isclose(np.dot(qut_i, qut_i), 1.0)

        for j in range(n_samples):
            if j == i:
                continue

            pos_j = np.array([states[j][0][k] for k in range(3)])
            qut_j = np.array([states[j][1].x, states[j][1].y, states[j][1].z, states[j][1].w])
            vel_j = np.array([states[j][2][k] for k in range(3)])
            omg_j = np.array([states[j][3][k] for k in range(3)])

            assert not np.allclose(pos_i, pos_j)
            assert not np.allclose(qut_i, qut_j)
            assert not np.allclose(vel_i, vel_j)
            assert not np.allclose(omg_i, omg_j)

if __name__ == "__main__":
    # test_QuadrotorPyBulletStatePropagator_propagate_drift(None)
    # test_QuadrotorPyBulletStatePropagator_propagate_path_climb(None)
    # test_QuadrotorPyBulletSetup_observeLidar_2(None)
    # test_QuadrotorPyBulletSetup_observeState_1()
    test_QuadrotorPyBulletSetup_state_sampler_0(None)