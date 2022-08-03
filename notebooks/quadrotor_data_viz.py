import torch
import time
import random
import numpy as np
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

# load in environment
# pbObstacleIds = pb.loadSDF("outputs/2022-07-25/13-44-40/pcg_room_009.world")
# pbWallIds = pb.loadSDF("outputs/2022-07-25/13-44-40/pcg_room_009_walls/model.sdf")
pbObstacleIds = pb.loadSDF("outputs/2022-08-02/13-14-50/pcg_room_000.world")
pbWallIds = pb.loadSDF("outputs/2022-08-02/13-14-50/pcg_room_000_walls/model.sdf")
pbFloorId = pb.loadURDF("plane100.urdf")

# recolor obstacles to make quadrotor more visible
for oid in pbObstacleIds:
    pb.changeVisualShape(objectUniqueId=oid, linkIndex=-1, rgbaColor=[1,1,1,0.5])

for wid in pbWallIds:
    pb.changeVisualShape(objectUniqueId=wid, linkIndex=-1, rgbaColor=[0,0,0,0.5])

# load in quadrotor risk data
# data = torch.load("outputs/2022-08-01/12-46-50/datagen_quadrotor_39b8d_e76ac_pcg_room_009.pt")
data = torch.load("outputs/2022-08-02/14-57-07/datagen_quadrotor_776cc_0f27f_pcg_room_000.pt")
n_data = len(data)
state_samples, risk_metrics, observations = tuple(zip(*data))

# populate with quadrotor urdfs at state randomly drawn from data set
n_viz_samples = 10
pbVizIds = n_viz_samples*[None]
dataSamples = random.sample(range(n_data), n_viz_samples)
for i in range(n_viz_samples):
    pbVizIds[i] = pb.loadURDF(str(Path(U.get_repo_path()).joinpath("tests/quadrotor.urdf")))
    copy_state_ompl2pb(pbBodyId=pbVizIds[i], pbClientId=pbClientId, omplState=state_samples[dataSamples[i]])


    # move camera to inspect each sampled state
    pb.resetDebugVisualizerCamera(
        cameraDistance=5.0,
        cameraYaw=50.0,
        cameraPitch=-35.0,
        cameraTargetPosition=[state_samples[dataSamples[i]][0][j] for j in range(3)]
    )

    #change color based on risk metric
    rcol = risk_metrics[dataSamples[i]]
    gcol = 1 - rcol
    pb.changeVisualShape(objectUniqueId=pbVizIds[i], linkIndex=-1, rgbaColor=[rcol,gcol,0,1])

    input("Press Enter to continue...")

# pi_ri_bu__bu = [(0.,0.,0.)]
# pi_rf_bu__bu = [(0.,0.,-100.)]
# while pb.isConnected():
#     ray_casts = pb.rayTestBatch(
#             rayFromPositions = pi_ri_bu__bu,
#             rayToPositions = pi_rf_bu__bu,
#             parentObjectUniqueId = pbQuadBodyId,
#             parentLinkIndex = -1,
#             physicsClientId = pbClientId)  
#     print("Raycast001: ",ray_casts[0][2]*100)
#     pb.stepSimulation()
#     time.sleep(dt)


if __name__ == "__main__":
    pass