import pybullet as pb
import time

# connect to GUI-based physics engine. Use pb.DIRECT for headless
pb.connect(pb.GUI)
dt = pb.getPhysicsEngineParameters()['fixedTimeStep']

# create vehicle and buildings collsion shape
veh_col_shape_id = pb.createCollisionShape(
    shapeType=pb.GEOM_MESH,
    fileName="meshes/base_link.stl",
    flags=pb.URDF_INITIALIZE_SAT_FEATURES #|p.GEOM_FORCE_CONCAVE_TRIMESH should only be used with fixed (mass=0) objects!
)
bld_col_shape_id = pb.createCollisionShape(
    shapeType=pb.GEOM_MESH,
    fileName="meshes/20220407_citymapgen_000/buildings.stl"
)

# create vehicle and buildings visual shape
veh_viz_shape_id = pb.createVisualShape(
    shapeType=pb.GEOM_MESH,
    fileName="meshes/base_link.stl",
)
bld_viz_shape_id = pb.createVisualShape(
    shapeType=pb.GEOM_MESH,
    fileName="meshes/20220407_citymapgen_000/buildings.stl"
)

# populate vehicle as body in world
veh_body_id = pb.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=veh_col_shape_id,
    baseVisualShapeIndex=veh_viz_shape_id,
    basePosition=(0, 0, 0),
    baseOrientation=(0, 0, 0, 1),
)
bld_body_id = pb.createMultiBody(
    baseMass=1000,
    baseCollisionShapeIndex=bld_col_shape_id,
    baseVisualShapeIndex=bld_viz_shape_id,
    basePosition=(0,0,1),
    baseOrientation=(0,0,0,1)
)


while pb.isConnected():
  pb.stepSimulation()
  time.sleep(dt)