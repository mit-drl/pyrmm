from ompl import base as ob
from ompl import control as oc
sspace = ob.RealVectorStateSpace(1)
cspace = oc.RealVectorControlSpace(sspace, 1)
c = cspace.allocControl()
print(type(c))
print(c[0])
