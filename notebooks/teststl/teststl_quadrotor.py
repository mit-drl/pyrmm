# Ref: https://github.com/bulletphysics/bullet3/issues/1293

import pybullet as pb
import pybullet_data as pbd
import time

# connect to GUI-based physics engine. Use pb.DIRECT for headless
pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pbd.getDataPath())
dt = pb.getPhysicsEngineParameters()['fixedTimeStep']

# populate vehicle and surroundings as body in world
quad_body_id = pb.loadURDF("meshes/quadrotor.urdf")
bld_body_id = pb.loadURDF("samurai.urdf")

# check for collisions between objects
print("Raycast001: ", pb.rayTest(rayFromPosition=(-1,0,-10), rayToPosition=(1,0,-10)))

# enable gravity
pb.setGravity(0,0,-1.0)

while pb.isConnected():
  pb.stepSimulation()
  pos, quat = pb.getBasePositionAndOrientation(quad_body_id)
  print("pos: {}, quat: {}".format(pos, quat))
  print("euler angles: {}".format(pb.getEulerFromQuaternion(quat)))
  time.sleep(dt)