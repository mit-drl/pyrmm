# Ref: https://github.com/bulletphysics/bullet3/issues/1293

import pybullet as pb
import time

# connect to GUI-based physics engine. Use pb.DIRECT for headless
pb.connect(pb.GUI)
dt = pb.getPhysicsEngineParameters()['fixedTimeStep']

# create vehicle and buildings collsion shape
# quad_col_shape_id = pb.createCollisionShape(
#     shapeType=pb.GEOM_MESH,
#     fileName="meshes/base_link.obj",
#     flags=pb.URDF_INITIALIZE_SAT_FEATURES
# )
veh_col_shape_id = pb.createCollisionShape(
    shapeType=pb.GEOM_MESH,
    fileName="meshes/base_link.stl",
    flags=pb.URDF_INITIALIZE_SAT_FEATURES #|p.GEOM_FORCE_CONCAVE_TRIMESH should only be used with fixed (mass=0) objects!
)
bld_col_shape_id = pb.createCollisionShape(
    shapeType=pb.GEOM_MESH,
    fileName="meshes/20220407_citymapgen_000/buildings.stl",
    flags=pb.GEOM_FORCE_CONCAVE_TRIMESH #should only be used with fixed (mass=0) objects!
)

# create vehicle and buildings visual shape
veh_viz_shape_id = pb.createVisualShape(
    shapeType=pb.GEOM_MESH,
    fileName="meshes/base_link.stl",
)
# quad_viz_shape_id = pb.createVisualShape(
#     shapeType=pb.GEOM_MESH,
#     fileName="meshes/quad.obj"
# )
bld_viz_shape_id = pb.createVisualShape(
    shapeType=pb.GEOM_MESH,
    fileName="meshes/20220407_citymapgen_000/buildings.stl"
)

# populate vehicle as body in world
veh_body_id = pb.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=veh_col_shape_id,
    baseVisualShapeIndex=veh_viz_shape_id,
    basePosition=(-1, 0, 0),
    baseOrientation=(0, 0, 0, 1),
)
# quad_body_id = pb.createMultiBody(
#     baseMass=1,
#     baseCollisionShapeIndex=quad_col_shape_id,
#     baseVisualShapeIndex=quad_viz_shape_id,
#     basePosition=(0,0,0),
#     baseOrientation=(0,0,0,1)
# )
quad_body_id = pb.loadURDF("meshes/quadrotor.urdf")
bld_body_id = pb.createMultiBody(
    baseMass=0, # static, immovable. Ref: https://usermanual.wiki/Document/pybullet20quickstart20guide.479068914.pdf
    baseCollisionShapeIndex=bld_col_shape_id,
    baseVisualShapeIndex=bld_viz_shape_id,
    basePosition=(0,0,0),
    baseOrientation=(0,0,0,1)
)

# check for collisions between objects
print("Vehicle AABB Box: ",pb.getAABB(bodyUniqueId=veh_body_id))
print("Raycast001: ", pb.rayTest(rayFromPosition=(-1,0,-10), rayToPosition=(1,0,-10)))


while pb.isConnected():
  pb.stepSimulation()
  time.sleep(dt)