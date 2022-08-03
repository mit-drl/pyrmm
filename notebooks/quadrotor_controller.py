# Visualize behavior of quadrotor controllers
import time
import numpy as np
import pybullet as pb
import pybullet_data as pbd

from pathlib import Path
from copy import deepcopy

import pyrmm.utils.utils as U
from pyrmm.setups.quadrotor import proportional_quaternion_controller_0, proportional_angular_rate_controller_0

pbClientId = pb.connect(pb.GUI)
dt = pb.getPhysicsEngineParameters()['fixedTimeStep']
pb.setAdditionalSearchPath(pbd.getDataPath())

# load in environment
pbFloorId = pb.loadURDF("plane100.urdf")

# Load in quadrotor model
pbQuadId = pb.loadURDF(str(Path(U.get_repo_path()).joinpath("tests/quadrotor.urdf")))
pos0, quat0 = pb.getBasePositionAndOrientation(pbQuadId)
vel0, omg0 = pb.getBaseVelocity(pbQuadId)

# reposition quadrotor in undesirable state
roll1 = 0
pitch1 = np.pi/2
yaw1 = 0
quat1 = pb.getQuaternionFromEuler([roll1, pitch1, yaw1])
pos1 = list(pos0)
pos1[2] = 5
pb.resetBasePositionAndOrientation(pbQuadId, pos1, quat1)


quat_sp = (0,0,0,1)
kp_quat = 1
kp_omg = (1,1,1)
while True:
    # observe state
    pos, quat = pb.getBasePositionAndOrientation(pbQuadId)
    vel, omg = pb.getBaseVelocity(pbQuadId)
    
    # compute control
    omg_sp = proportional_quaternion_controller_0(quat, quat_sp, kp_quat)
    moments = proportional_angular_rate_controller_0(omg, omg_sp, kp_omg)

    # apply control 
    pb.applyExternalTorque(objectUniqueId=pbQuadId,
                              linkIndex=-1,
                              torqueObj=moments,
                              flags=pb.LINK_FRAME,
                              physicsClientId=pbClientId
                              )
    pb.stepSimulation()
    
    inval = input('Continue time stepping?')
    if inval.lower() in ['n', 'no']:
        break

pb.disconnect()

