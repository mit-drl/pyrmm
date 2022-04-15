import pathlib
import numpy as np
import pybullet as pb

import pyrmm.dynamics.quadrotor as QD

from copy import deepcopy

QUAD_URDF = str(pathlib.Path(__file__).parent.absolute().joinpath("quadrotor.urdf"))

def test_copy_state_0():
    '''test that a state is not modified after a pass between ompl and pb'''

    # ~~ ARRANGE ~~

    # connect to headless physics engine
    pbClientId = pb.connect(pb.DIRECT)

    # create pybullet instance of quadrotor
    pbQuadBodyId = pb.loadURDF(QUAD_URDF)

    # store pb state for later comparison
    old_pos, old_quat = deepcopy(pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId))
    old_vel, old_angv = deepcopy(pb.getBaseVelocity(pbQuadBodyId, pbClientId))

    # create ompl state space and state
    omplStateSpace = QD.QuadrotorStateSpace()
    omplState = omplStateSpace.allocState()

    # ~~ ACT ~~
    # copy pb state to ompl
    QD.copy_state_pb2ompl(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # copy ompl state back to pb
    QD.copy_state_ompl2pb(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # ~~ ASSERT ~~
    pos, quat = pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId)
    vel, angv = pb.getBaseVelocity(pbQuadBodyId, pbClientId)
    assert np.allclose(pos, old_pos)
    assert np.allclose(quat, old_quat)
    assert np.allclose(vel, old_vel)
    assert np.allclose(angv, old_angv)

    # disconnect from pybullet physics client
    pb.disconnect()

def test_copy_state_1():
    '''test that a state that is modified is not equal to original state'''

    # ~~ ARRANGE ~~

    # connect to headless physics engine
    pbClientId = pb.connect(pb.DIRECT)

    # create pybullet instance of quadrotor
    pbQuadBodyId = pb.loadURDF(QUAD_URDF)

    # store pb state for later comparison
    old_pos, old_quat = deepcopy(pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId))
    old_vel, old_angv = deepcopy(pb.getBaseVelocity(pbQuadBodyId, pbClientId))

    # create ompl state space and state
    omplStateSpace = QD.QuadrotorStateSpace()
    omplState = omplStateSpace.allocState()

    # ~~ ACT ~~
    # copy pb state to ompl
    QD.copy_state_pb2ompl(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # modify the quaternion
    omplState[1].setAxisAngle(1,0,0,np.pi/6)

    # copy ompl state back to pb
    QD.copy_state_ompl2pb(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # ~~ ASSERT ~~
    pos, quat = pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId)
    vel, angv = pb.getBaseVelocity(pbQuadBodyId, pbClientId)
    assert np.allclose(pos, old_pos)
    assert not np.allclose(quat, old_quat)
    assert np.allclose(vel, old_vel)
    assert np.allclose(angv, old_angv)

    # disconnect from pybullet physics client
    pb.disconnect()