import pytest
import pathlib
import numpy as np
import pybullet as pb

import pyrmm.dynamics.quadrotor as QD

from copy import deepcopy

QUAD_URDF = str(pathlib.Path(__file__).parent.absolute().joinpath("quadrotor.urdf"))

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


def test_copy_state_1(pybullet_copy_state):
    '''test that a state that is modified is not equal to original state'''

    # ~~ ARRANGE ~~
    pbClientId, pbQuadBodyId, omplState = pybullet_copy_state

    # store pb state for later comparison
    old_pos, old_quat = deepcopy(pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId))
    old_vel, old_angv = deepcopy(pb.getBaseVelocity(pbQuadBodyId, pbClientId))

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
    QD.copy_state_pb2ompl(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # copy ompl state back to pb
    QD.copy_state_ompl2pb(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # ~~ ASSERT ~~
    pos, quat = pb.getBasePositionAndOrientation(pbQuadBodyId, pbClientId)
    vel, angv = pb.getBaseVelocity(pbQuadBodyId, pbClientId)
    assert np.allclose(pos, pos0)
    assert np.allclose(quat, quat0)
    assert np.allclose(vel, vel0)
    assert np.allclose(angv, angv0)

if __name__ == "__main__":
    test_copy_state_2(None)