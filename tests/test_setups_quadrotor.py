import pathlib
import copy
import numpy as np
import pybullet as pb

from ompl import control as oc

import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD
from pyrmm.setups.quadrotor import QuadrotorPyBulletStatePropagator

QUAD_URDF = str(pathlib.Path(__file__).parent.absolute().joinpath("quadrotor.urdf"))

def test_QuadrotorPyBulletStatePropagator_propagate_0():
    '''Basic test that a known thrust moves the body an expected amount'''

    # ~~ ARRANGE ~~

    # connect to headless physics engine
    pbClientId = pb.connect(pb.DIRECT)

    # create pybullet instance of quadrotor
    pbQuadBodyId = pb.loadURDF(QUAD_URDF)

    # create OMPL space information object
    sspace = QD.QuadrotorStateSpace()
    cspace = QD.QuadrotorThrustMomentControlSpace(stateSpace=sspace)
    space_info = oc.SpaceInformation(stateSpace=sspace, controlSpace=cspace)

    # create state propagator
    quadPropagator = QuadrotorPyBulletStatePropagator(
        pbBodyId=pbQuadBodyId, 
        pbClientId=pbClientId, 
        spaceInformation=space_info)

    # create initial and resulting OMPL state objects
    init_state = space_info.getStateSpace().allocState()
    QD.copy_state_pb2ompl(pbQuadBodyId, pbClientId, init_state)
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

    # ~~ CLEANUP ~~
    # disconnect from pybullet physics client
    pb.disconnect()

if __name__ == "__main__":
    test_QuadrotorPyBulletStatePropagator_propagate_0()