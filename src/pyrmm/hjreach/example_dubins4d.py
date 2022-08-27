import numpy as np
import math

import heterocl as hcl

from scipy.interpolate import interpn

# Utility functions to initialize the problem
from odp.Grid import Grid
from odp.Shapes import CylinderShape

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCar4D

# Plot options
from odp.Plots import PlotOptions

# Solver core
from odp.solver import HJSolver

from odp.spatialDerivatives.first_orderENO4D import spa_derivX1_4d, spa_derivX2_4d, spa_derivX3_4d, spa_derivX4_4d


# Discretize the state space over which PDE is solved
g = Grid(np.array([-4.0, -4.0, 0.0, -math.pi]), np.array([4.0, 4.0, 4.0, math.pi]), 4, np.array([60, 60, 20, 36]), [3])

# Reachable set
# NOTE: Should be a sphere in x,y but I don't yet fully understand how to interpret 
# CylinderShape API
goal = CylinderShape(g, [2,3], np.zeros(4), 0.5)

# Avoid set
obstacle = CylinderShape(g, [2,3], np.array([1.0, 1.0, 0.0, 0.0]), 0.5)

# Look-back length and time step
lookback_length = 0.25
t_step = 0.05
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

# dynamics
my_car = DubinsCar4D(uMode="min", dMode="max", dMin = [0.0,0.0], dMax = [0.0,0.0])

po2 = PlotOptions(do_plot=True, plot_type="3d_plot", plotDims=[0,1,3],
                  slicesCut=[19])

"""
Assign one of the following strings to `TargetSetMode` to specify the characteristics of computation
"TargetSetMode":
{
"none" -> compute Backward Reachable Set, 
"minVWithV0" -> min V with V0 (compute Backward Reachable Tube),
"maxVWithV0" -> max V with V0,
"maxVWithVInit" -> compute max V over time,
"minVWithVInit" -> compute min V over time,
"minVWithVTarget" -> min V with target set (if target set is different from initial V0)
"maxVWithVTarget" -> max V with target set (if target set is different from initial V0)
}

(optional)
Please specify this mode if you would like to add another target set, which can be an obstacle set
for solving a reach-avoid problem
"ObstacleSetMode":
{
"minVWithObstacle" -> min with obstacle set,
"maxVWithObstacle" -> max with obstacle set
}
"""

compMethods = { "TargetSetMode": "minVWithVTarget",
                "ObstacleSetMode": "maxVWithObstacle"}

# HJSolver(dynamics object, grid, initial value function, time length, system objectives, plotting options)
result = HJSolver(my_car, g, [goal, obstacle], tau, compMethods, po2, saveAllTimeSteps=True )
V = result[..., 0]
# V_hcl = hcl.Tensor(V.shape, buf=hcl.asarray(V), dtype='float32')

# Compute spacial derivatives
def avg_spa_derivX1_4d(V, i, j, k, l):
    dV_dx1_L = hcl.scalar(0, "dV_dx1_L")
    dV_dx1_R = hcl.scalar(0, "dV_dx1_R")
    dV_dx1_L[0], dV_dx1_R[0] = spa_derivX1_4d(i, j, k, l, V, g)
    return (dV_dx1_L + dV_dx1_R) / 2
def avg_spa_derivX2_4d(V, i, j, k, l):
    dV_dx2_L = hcl.scalar(0, "dV_dx2_L")
    dV_dx2_R = hcl.scalar(0, "dV_dx2_R")
    dV_dx2_L[0], dV_dx2_R[0] = spa_derivX2_4d(i, j, k, l, V, g)
    return (dV_dx2_L + dV_dx2_R) / 2
def avg_spa_derivX3_4d(V, i, j, k, l):
    dV_dx3_L = hcl.scalar(0, "dV_dx3_L")
    dV_dx3_R = hcl.scalar(0, "dV_dx3_R")
    dV_dx3_L[0], dV_dx3_R[0] = spa_derivX3_4d(i, j, k, l, V, g)
    return (dV_dx3_L + dV_dx3_R) / 2
def avg_spa_derivX4_4d(V, i, j, k, l):
    dV_dx4_L = hcl.scalar(0, "dV_dx4_L")
    dV_dx4_R = hcl.scalar(0, "dV_dx4_R")
    dV_dx4_L[0], dV_dx4_R[0] = spa_derivX4_4d(i, j, k, l, V, g)
    return (dV_dx4_L + dV_dx4_R) / 2
def compute_spa_deriv_4d(V):
    dV_dx1 = hcl.compute(V.shape, lambda i, j, k, l: avg_spa_derivX1_4d(V,i,j,k,l), dtype=hcl.Float())
    dV_dx2 = hcl.compute(V.shape, lambda i, j, k, l: avg_spa_derivX2_4d(V,i,j,k,l), dtype=hcl.Float())
    dV_dx3 = hcl.compute(V.shape, lambda i, j, k, l: avg_spa_derivX3_4d(V,i,j,k,l), dtype=hcl.Float())
    dV_dx4 = hcl.compute(V.shape, lambda i, j, k, l: avg_spa_derivX4_4d(V,i,j,k,l), dtype=hcl.Float())
    return dV_dx1, dV_dx2, dV_dx3, dV_dx4
# def compute_opt_ctrl(V):
#     dV_dx1, dV_dx2, dV_dx3, dV_dx4 = compute_spa_deriv_4d(V)
#     opt_ctrl = hcl.compute(
#         V.shape, 
#         lambda i, j, k, l: my_car.opt_ctrl(None, None, (dV_dx1[i,j,k,l], dV_dx2[i,j,k,l], dV_dx3[i,j,k,l], dV_dx4[i,j,k,l])))
#     return opt_ctrl
def get_opt_ctrl_at_pt(V,i,j,k,l):
    dV_dx1_pt = avg_spa_derivX1_4d(V,i,j,k,l)
    dV_dx2_pt = avg_spa_derivX2_4d(V,i,j,k,l)
    dV_dx3_pt = avg_spa_derivX3_4d(V,i,j,k,l)
    dV_dx4_pt = avg_spa_derivX4_4d(V,i,j,k,l)
    return my_car.opt_ctrl(None, None, (dV_dx1_pt[0], dV_dx2_pt[0], dV_dx3_pt[0], dV_dx4_pt[0]))[0]
def compute_opt_ctrl(V):
    opt_ctrl = hcl.compute(V.shape, lambda i,j,k,l: get_opt_ctrl_at_pt(V,i,j,k,l), dtype=hcl.Float())
    return opt_ctrl

# define inputs
hclph_V = hcl.placeholder(V.shape, dtype=hcl.Float())

# create scheduler
spa_deriv_sched = hcl.create_schedule([hclph_V], compute_spa_deriv_4d)
# sched = hcl.create_schedule([hclph_V], compute_opt_ctrl)

# create executable function
compute_spa_derivX1_4d_func = hcl.build(schedule=spa_deriv_sched)
# compute_opt_ctrl_func = hcl.build(schedule=sched)

# prepare inputs and outputs
hcl_V = hcl.asarray(V)
hcl_dV_dx1 = hcl.asarray(np.zeros(V.shape))
hcl_dV_dx2 = hcl.asarray(np.zeros(V.shape))
hcl_dV_dx3 = hcl.asarray(np.zeros(V.shape))
hcl_dV_dx4 = hcl.asarray(np.zeros(V.shape))
# hcl_opt_ctrl_u1 = hcl.asarray(np.zeros(V.shape))
# hcl_opt_ctrl_u2 = hcl.asarray(np.zeros(V.shape))

# run executable
compute_spa_derivX1_4d_func(hcl_V, hcl_dV_dx1, hcl_dV_dx2, hcl_dV_dx3, hcl_dV_dx4)
# compute_opt_ctrl_func(hcl_V, hcl_opt_ctrl_u1, hcl_opt_ctrl_u2)

# convert to numpy
dV_dx1 = hcl_dV_dx1.asnumpy()
dV_dx2 = hcl_dV_dx2.asnumpy()
dV_dx3 = hcl_dV_dx3.asnumpy()
dV_dx4 = hcl_dV_dx4.asnumpy()
# opt_ctrl_u1 = hcl_opt_ctrl_u1.asnumpy()
# opt_ctrl_u2 = hcl_opt_ctrl_u2.asnumpy()

# sched = hcl.create_schedule([hclph_V, hclph_dV_dx1, hclph_dV_dx2, hclph_dV_dx3, hclph_dV_dx4], compute_spatial_deriv)
# compute_spatial_deriv_func = hcl.build(schedule=sched)
# compute_spatial_deriv_func(hcl.asarray(V), dV_dx1, dV_dx2, dV_dx3, dV_dx4)

# compute V at arbitrary location
x0 = -0.5
y0 = -0.5
v0 = 1.0
th0 = math.pi/4
X0 = [x0,y0,v0,th0]
V_X0 = interpn(g.grid_points, V, X0)
print("Value function V at X={}: {}".format(X0, V_X0))

# Compute spacial derivatives at location
dV_dx1_X0 = interpn(g.grid_points, dV_dx1, X0)
dV_dx2_X0 = interpn(g.grid_points, dV_dx2, X0)
dV_dx3_X0 = interpn(g.grid_points, dV_dx3, X0)
dV_dx4_X0 = interpn(g.grid_points, dV_dx4, X0)
print("Spacial Derivative of value function V at X0={}: ({},{},{},{})".format(X0, dV_dx1_X0, dV_dx2_X0, dV_dx3_X0, dV_dx4_X0))

# # define inputs
# # hclph_dV_dx1_X0 = hcl.placeholder((), "hclph_dV_dx1_X0")
# # hclph_dV_dx2_X0 = hcl.placeholder((), "hclph_dV_dx2_X0")
# # hclph_dV_dx3_X0 = hcl.placeholder((), "hclph_dV_dx3_X0")
# # hclph_dV_dx4_X0 = hcl.placeholder((), "hclph_dV_dx4_X0")

# # create scheduler
# # opt_ctrl_sched = hcl.create_schedule([hclph_dV_dx1_X0, hclph_dV_dx2_X0, hclph_dV_dx3_X0, hclph_dV_dx4_X0], compute_opt_ctrl)
# opt_ctrl_sched = hcl.create_schedule([hclph_V], compute_opt_ctrl)

# # create executable function
# compute_opt_ctrl_func = hcl.build(schedule=opt_ctrl_sched)

# # prepare inputs and outputs
# # hcl_opt_ctrl_X0 = hcl.asarray(np.zeros(4,))
# hcl_opt_ctrl_u1 = hcl.asarray(np.zeros(V.shape))

# # run executable
# compute_opt_ctrl_func(hcl_V, hcl_opt_ctrl_u1)

# # compute optimal control at location
# hcl_dV_dx1_X0 = hcl.scalar(0)
# hcl_dV_dx1_X0[0] = dV_dx1_X0
# opt_ctrl_X0 = my_car.opt_ctrl(
#     0, X0, 
#     (dV_dx1_X0, dV_dx2_X0, dV_dx3_X0, dV_dx4_X0))
# print("Control at X0={}: ({},{})".format(X0, opt_ctrl_X0))
# opt_ctrl_u1_X0 = interpn(g.grid_points, opt_ctrl_u1, X0)
# opt_ctrl_u2_X0 = interpn(g.grid_points, opt_ctrl_u2, X0)
# print("Control at X0={}: ({},{})".format(X0, opt_ctrl_u1_X0, opt_ctrl_u2_X0))

# get V at particular state
print("DONE!")