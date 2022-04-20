import pytest
import pathlib
import copy
import numpy as np
import pybullet as pb

from ompl import control as oc

import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD
from pyrmm.setups.quadrotor import QuadrotorPyBulletStatePropagator

QUAD_URDF = str(pathlib.Path(__file__).parent.absolute().joinpath("quadrotor.urdf"))

@pytest.fixture
def quadrotor_pybullet_propagator():
    # ~~ ARRANGE ~~

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

    yield space_info, quadPropagator

    # ~~ Teardown ~~
    # disconnect from pybullet physics client
    pb.disconnect()


def test_QuadrotorPyBulletStatePropagator_propagate_0(quadrotor_pybullet_propagator):
    '''Basic test that a known thrust moves the body an expected amount'''

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

if __name__ == "__main__":
    test_QuadrotorPyBulletStatePropagator_propagate_0()