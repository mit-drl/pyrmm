import pytest
import pybullet as pb
import pybullet_data as pbd

from pathlib import Path

QUAD_URDF = str(Path(__file__).parent.absolute().joinpath("quadrotor.urdf"))
PCG_ROOM = Path(__file__).parent.absolute().joinpath("pcg_room_test000.world")

def test_quadrotor_wall_contact_0():
    '''check that quadrotor does not contact walls'''
    # ~~~ ARRANGE ~~~

    # connect to pybullet physics server
    pbClientId = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pbd.getDataPath())

    # load pcg room, floor, and quadrotor

    # load environment
    pbObstacleIds = pb.loadSDF(str(PCG_ROOM))
    pbWallIds = pb.loadSDF(str(PCG_ROOM.with_suffix('')) + '_walls/model.sdf')
    pbFloorId = pb.loadURDF("plane100.urdf")
    pbQuadBodyId = pb.loadURDF(QUAD_URDF)

    # move quadrotor to non-contacting position
    pb.resetBasePositionAndOrientation(
        bodyUniqueId=pbQuadBodyId,
        posObj=[0.0, 0.0, 1.5],
        ornObj=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=pbClientId
    )

    # ~~~ ACT ~~~

    # call collision detection function to assess collisions 
    # independent of simulation stepping
    # ref: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
    pb.performCollisionDetection()

    # get all contacts between quadrotor and other objects in env
    contacts = pb.getContactPoints(
        bodyA = pbQuadBodyId,
        physicsClientId = pbClientId
    )

    # ~~~ ASSERT ~~~
    assert len(contacts) == 0

    # ~~~ TEARDOWN ~~~
    pb.disconnect(pbClientId)

if __name__ == "__main__":
    test_quadrotor_wall_contact_0()