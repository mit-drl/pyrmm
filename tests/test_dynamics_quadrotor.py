import pytest
import pathlib
import pickle
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
    QD.copy_state_pb2ompl(pbClientId=pbClientId, pbBodyId=pbQuadBodyId, omplState=omplState)

    # ~~ ACT ~~
    # update pickler
    QD.update_pickler_quadrotorstate()

    # pickle and unpickle ompl state
    omplState_copy = pickle.loads(pickle.dumps(omplState))

    # ~~ ASSERT ~~
    assert np.allclose(pos0, [omplState_copy[0][i] for i in range(3)])
    qc = omplState_copy[1]
    assert np.allclose(quat0, [qc.x, qc.y, qc.z, qc.w])
    assert np.allclose(vel0, [omplState_copy[2][i] for i in range(3)])
    assert np.allclose(angv0, [omplState_copy[3][i] for i in range(3)])

def test_QuadrotorStateSpace_bounds_0():
    '''check that state space bounds are set as intended'''
    # ~~ ARRANGE ~~
    # random yet fixed position, velocity, and ang vel bounds
    bounds = dict()
    bounds['pos_low']= [-2.55944188, -3.08836728, -2.58956761]
    bounds['pos_high'] = [13.58335454, 11.78403468,  5.68640255]
    bounds['vel_low']= [-4.34878973,  0.99390659,  1.31070602]
    bounds['vel_high'] = [6.0623723 , 8.94204147, 9.349446  ]
    bounds['omg_low']= [-0.85770383,  4.52460971, -2.74973743]
    bounds['omg_high'] = [11.15227062,  9.96662472, 10.04022507]

    # ~~ ACT ~~
    # create state space with bounds
    quadspace = QD.QuadrotorStateSpace(bounds=bounds)

    # ~~ ASSERT ~~~
    pos_bounds = quadspace.getSubspace(0).getBounds()
    assert np.isclose(pos_bounds.low[0], bounds['pos_low'][0])
    assert np.isclose(pos_bounds.low[1], bounds['pos_low'][1])
    assert np.isclose(pos_bounds.low[2], bounds['pos_low'][2])
    assert np.isclose(pos_bounds.high[0], bounds['pos_high'][0])
    assert np.isclose(pos_bounds.high[1], bounds['pos_high'][1])
    assert np.isclose(pos_bounds.high[2], bounds['pos_high'][2])

    vel_bounds = quadspace.getSubspace(2).getBounds()
    assert np.isclose(vel_bounds.low[0], bounds['vel_low'][0])
    assert np.isclose(vel_bounds.low[1], bounds['vel_low'][1])
    assert np.isclose(vel_bounds.low[2], bounds['vel_low'][2])
    assert np.isclose(vel_bounds.high[0], bounds['vel_high'][0])
    assert np.isclose(vel_bounds.high[1], bounds['vel_high'][1])
    assert np.isclose(vel_bounds.high[2], bounds['vel_high'][2])

    omg_bounds = quadspace.getSubspace(3).getBounds()
    assert np.isclose(omg_bounds.low[0], bounds['omg_low'][0])
    assert np.isclose(omg_bounds.low[1], bounds['omg_low'][1])
    assert np.isclose(omg_bounds.low[2], bounds['omg_low'][2])
    assert np.isclose(omg_bounds.high[0], bounds['omg_high'][0])
    assert np.isclose(omg_bounds.high[1], bounds['omg_high'][1])
    assert np.isclose(omg_bounds.high[2], bounds['omg_high'][2])
    
def test_QuadrotorStateSpace_bounds_sample_0():
    '''check that state space with zero-volume bounds sample a deterministic number'''
    # ~~ ARRANGE ~~
    # random yet fixed zero-volume bounds
    bounds = dict()
    bounds['pos_low']= [-99.68020119,  53.3656405 ,  77.48463094]
    bounds['pos_high'] = [-99.68020119,  53.3656405 ,  77.48463094]
    bounds['vel_low']= [-5.74721878,  4.86715528, -5.42107527]
    bounds['vel_high'] = [-5.74721878,  4.86715528, -5.42107527]
    bounds['omg_low']= [ 6.75674113,  7.69125197, -4.14222264]
    bounds['omg_high'] = [ 6.75674113,  7.69125197, -4.14222264]

    # ~~ ACT ~~
    # create state space with bounds
    quadspace = QD.QuadrotorStateSpace(bounds=bounds)
    ssampler = quadspace.allocStateSampler()
    sstate = quadspace.allocState()
    ssampler.sampleUniform(sstate)

    # ~~ ASSERT ~~~
    assert np.isclose(sstate[0][0], bounds['pos_low'][0])
    assert np.isclose(sstate[0][1], bounds['pos_low'][1])
    assert np.isclose(sstate[0][2], bounds['pos_low'][2])
    assert np.isclose(sstate[2][0], bounds['vel_low'][0])
    assert np.isclose(sstate[2][1], bounds['vel_low'][1])
    assert np.isclose(sstate[2][2], bounds['vel_low'][2])
    assert np.isclose(sstate[3][0], bounds['omg_low'][0])
    assert np.isclose(sstate[3][1], bounds['omg_low'][1])
    assert np.isclose(sstate[3][2], bounds['omg_low'][2])
    q = np.array([sstate[1].x, sstate[1].y, sstate[1].z, sstate[1].w])
    assert np.isclose(np.dot(q,q), 1.0)

if __name__ == "__main__":
    # test_copy_state_2(None)
    test_update_pickler_quadrotorstate_0(None)