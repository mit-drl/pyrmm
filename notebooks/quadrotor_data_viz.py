import torch
import time
import pybullet as pb
import pybullet_data as pbd

from copy import deepcopy
from pathlib import Path

import pyrmm.utils.utils as U
from pyrmm.setups.quadrotor import copy_state_ompl2pb, update_pickler_quadrotorstate

pbClientId = pb.connect(pb.GUI)
dt = pb.getPhysicsEngineParameters()['fixedTimeStep']
pb.setAdditionalSearchPath(pbd.getDataPath())

# enabling copying quadrotor state
update_pickler_quadrotorstate()

# load in quadrotor risk data
data = torch.load("outputs/2022-08-01/12-46-50/datagen_quadrotor_39b8d_e76ac_pcg_room_009.pt")
state_samples, risk_metrics, observations = tuple(zip(*data))

# populate with quadrotor urdf
pbQuadBodyId = pb.loadURDF(str(Path(U.get_repo_path()).joinpath("tests/quadrotor.urdf")))
copy_state_ompl2pb(pbBodyId=pbQuadBodyId, pbClientId=pbClientId, omplState=state_samples[0])
p_bu_wu__wu = [state_samples[0][0][i] for i in range(3)]
p_bu_wu__wu[1] = 7.0
q_bu_wu = [state_samples[0][1].x, state_samples[0][1].y, state_samples[0][1].z, state_samples[0][1].w]
pb.resetBasePositionAndOrientation(
        bodyUniqueId=pbQuadBodyId,
        posObj=p_bu_wu__wu,
        ornObj=q_bu_wu,
        physicsClientId=pbClientId
    )
pb.resetBaseVelocity(
        objectUniqueId=pbQuadBodyId,
        linearVelocity=[0,0,0],
        angularVelocity=[0,0,0],
        physicsClientId=pbClientId
    )

# load in environment
pbObstacleIds = pb.loadSDF("outputs/2022-07-25/13-44-40/pcg_room_009.world")
pbWallIds = pb.loadSDF("outputs/2022-07-25/13-44-40/pcg_room_009_walls/model.sdf")
pbFloorId = pb.loadURDF("plane100.urdf")

# move camera
pb.resetDebugVisualizerCamera(
    cameraDistance=5.0,
    cameraYaw=50.0,
    cameraPitch=-35.0,
    cameraTargetPosition=[state_samples[0][0][i] for i in range(3)]
)

pi_ri_bu__bu = [(0.,0.,0.)]
pi_rf_bu__bu = [(0.,0.,-100.)]
while pb.isConnected():
    ray_casts = pb.rayTestBatch(
            rayFromPositions = pi_ri_bu__bu,
            rayToPositions = pi_rf_bu__bu,
            parentObjectUniqueId = pbQuadBodyId,
            parentLinkIndex = -1,
            physicsClientId = pbClientId)  
    print("Raycast001: ",ray_casts[0][2]*100)
    pb.stepSimulation()
    time.sleep(dt)


if __name__ == "__main__":
    pass