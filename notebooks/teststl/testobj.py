import pybullet as p
import time

p.connect(p.GUI)
dt = p.getPhysicsEngineParameters()['fixedTimeStep']
mesh_file = "meshes/qUwYeQXbgT.obj"

col_shape_id = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName=mesh_file,
    flags=p.URDF_INITIALIZE_SAT_FEATURES
)

viz_shape_id = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName=mesh_file,
)

body_id = p.createMultiBody(
    baseMass = 1,
    baseCollisionShapeIndex=col_shape_id,
    baseVisualShapeIndex=viz_shape_id,
    basePosition=(0, 0, 0),
    baseOrientation=(0, 0, 0, 1),
)


while p.isConnected():
  p.stepSimulation()
  time.sleep(dt)