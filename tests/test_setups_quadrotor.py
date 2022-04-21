import pytest
import pathlib
import copy
import numpy as np
import pybullet as pb
import pybullet_data as pbd

from ompl import control as oc

import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD
from pyrmm.setups.quadrotor import \
    QuadrotorPyBulletSetup, \
    QuadrotorPyBulletStatePropagator

QUAD_URDF = str(pathlib.Path(__file__).parent.absolute().joinpath("quadrotor.urdf"))

def get_quadrotor_pybullet_propagator_objects():
    # connect to headless physics engine
    pbClientId = pb.connect(pb.DIRECT)

    # create pybullet instance of quadrotor
    pbQuadBodyId = pb.loadURDF(QUAD_URDF)

    # create OMPL space information object
    sspace = QD.QuadrotorStateSpace()
    cspace = QD.QuadrotorThrustMomentControlSpace(
        stateSpace=sspace,
        fzmax=8.0,
        mxmax=2.0,
        mymax=2.0,
        mzmax=1.0)
    space_info = oc.SpaceInformation(stateSpace=sspace, controlSpace=cspace)

    # create state propagator
    quadPropagator = QuadrotorPyBulletStatePropagator(
        pbBodyId=pbQuadBodyId, 
        pbClientId=pbClientId, 
        spaceInformation=space_info)

    return space_info, quadPropagator

@pytest.fixture
def quadrotor_pybullet_propagator():
    # ~~ ARRANGE ~~

    space_info, quadPropagator = get_quadrotor_pybullet_propagator_objects()

    yield space_info, quadPropagator

    # ~~ TEARDOWN ~~
    # disconnect from pybullet physics client
    pb.disconnect()

@pytest.fixture
def quadrotor_pybullet_setup():
    # ~~ ARRANGE ~~
    qpbsetup = QuadrotorPyBulletSetup()

    # ~~ PASS TO TEST
    yield qpbsetup

    # ~~ Teardown ~~
    # disconnect from pybullet physics client
    pb.disconnect()


def test_QuadrotorPyBulletSetup_state_checker_0(quadrotor_pybullet_setup):
    '''test that simple, initial state in empty environment has no collisions'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup

    # get initial state from pybullet
    s0 = qpbsetup.space_info.allocState()
    QD.copy_state_pb2ompl(
        pbBodyId=qpbsetup.pbBodyId,
        pbClientId=qpbsetup.pbClientId,
        omplState=s0
    )

    # ~~ ACT ~~
    # get state validity
    is_valid = qpbsetup.space_info.isValid(s0)

    # ~~ ASSERT ~~
    assert is_valid

def test_QuadrotorPyBulletSetup_state_checker_1(quadrotor_pybullet_setup):
    '''test that collision with floor plane causes invalid state'''

    # ~~ ARRANGE ~~

    # get quadrotor setup object
    if quadrotor_pybullet_setup is None:
        qpbsetup = QuadrotorPyBulletSetup()
    else:
        qpbsetup = quadrotor_pybullet_setup

    # load in floor plane to environment (from pybullet_data)
    pb.setAdditionalSearchPath(pbd.getDataPath())
    floorBodyId = pb.loadURDF("plane100.urdf")

    # get initial state from pybullet
    s0 = qpbsetup.space_info.allocState()
    QD.copy_state_pb2ompl(
        pbBodyId=qpbsetup.pbBodyId,
        pbClientId=qpbsetup.pbClientId,
        omplState=s0
    )

    # ~~ ACT ~~
    # get state validity
    is_valid = qpbsetup.space_info.isValid(s0)

    # ~~ ASSERT ~~
    assert not is_valid

def test_QuadrotorPyBulletStatePropagator_propagate_hover(quadrotor_pybullet_propagator):
    '''Test that perfect hover thrust does not move the quadrotor'''

    # ~~ ARRANGE ~~

    # unpack fixture values
    space_info, quadPropagator = quadrotor_pybullet_propagator

    # create initial and resulting OMPL state objects
    init_state = space_info.getStateSpace().allocState()
    QD.copy_state_pb2ompl(quadPropagator.pbBodyId, quadPropagator.pbClientId, init_state)
    result_state = space_info.getStateSpace().allocState()

    # calculate hovering thrust
    mass = 0.5  # see quadrotor.urdf
    thrust = mass * U.GRAV_CONST
    control = space_info.getControlSpace().allocControl()
    control[0] = thrust
    control[1] = 0.0
    control[2] = 0.0
    control[3] = 0.0

    # ~~ ACT ~~

    # apply hovering thrust
    quadPropagator.propagate(
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

def test_QuadrotorPyBulletStatePropagator_propagate_drift(quadrotor_pybullet_propagator):
    '''Test that non-zero initial horizontal velocity drifts an known distance'''

    # ~~ ARRANGE ~~

    if quadrotor_pybullet_propagator is None:
        space_info, quadPropagator = get_quadrotor_pybullet_propagator_objects()
    else:
        space_info, quadPropagator = quadrotor_pybullet_propagator

    # set non-zero initial horizontal velocity
    xvel =  9.4810
    dur = 6.0445
    pb.resetBaseVelocity(
        objectUniqueId=quadPropagator.pbBodyId,
        linearVelocity=[xvel, 0, 0],
        angularVelocity=[0.0, 0.0, 0.0],
        physicsClientId=quadPropagator.pbClientId)

    # turn of linear damping for perfect drift
    pb.changeDynamics(
        bodyUniqueId=quadPropagator.pbBodyId,
        linkIndex=-1,
        linearDamping=0.0
    )

    # create initial and resulting OMPL state objects
    init_state = space_info.getStateSpace().allocState()
    QD.copy_state_pb2ompl(quadPropagator.pbBodyId, quadPropagator.pbClientId, init_state)
    result_state = space_info.getStateSpace().allocState()

    # calculate hovering thrust
    mass = 0.5  # see quadrotor.urdf
    thrust = mass * U.GRAV_CONST
    control = space_info.getControlSpace().allocControl()
    control[0] = thrust
    control[1] = 0.0
    control[2] = 0.0
    control[3] = 0.0

    # ~~ ACT ~~

    # apply hovering thrust
    quadPropagator.propagate(
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


def test_QuadrotorPyBulletStatePropagator_propagate_climb(quadrotor_pybullet_propagator):
    '''Test that known climb rate results in expect climb distance'''

    # ~~ ARRANGE ~~

    if quadrotor_pybullet_propagator is None:
        space_info, quadPropagator = get_quadrotor_pybullet_propagator_objects()
    else:
        space_info, quadPropagator = quadrotor_pybullet_propagator

    # create initial and resulting OMPL state objects
    init_state = space_info.getStateSpace().allocState()
    QD.copy_state_pb2ompl(quadPropagator.pbBodyId, quadPropagator.pbClientId, init_state)
    result_state = space_info.getStateSpace().allocState()

    # turn of linear damping for no velocity damping
    pb.changeDynamics(
        bodyUniqueId=quadPropagator.pbBodyId,
        linkIndex=-1,
        linearDamping=0.0
    )

    # calculate hovering thrust
    climb_acc = -0.122648
    dur = 4.18569
    mass = 0.5  # see quadrotor.urdf
    thrust = mass * (U.GRAV_CONST + climb_acc)
    control = space_info.getControlSpace().allocControl()
    control[0] = thrust
    control[1] = 0.0
    control[2] = 0.0
    control[3] = 0.0

    # ~~ ACT ~~

    # apply hovering thrust
    quadPropagator.propagate(
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

if __name__ == "__main__":
    test_QuadrotorPyBulletStatePropagator_propagate_drift(None)