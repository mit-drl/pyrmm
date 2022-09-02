import numpy as np
import math

import heterocl as hcl

from scipy.interpolate import interpn

from odp.Grid import Grid
from odp.Shapes import CylinderShape
from odp.dynamics import DubinsCar4D
from odp.solver import computeSpatDerivArray

from dubins4d_reachavoid_agent import HJReachDubins4dReachAvoidAgent

TIME_HORIZON = 1.0
TIME_STEP = 0.05
GRID_LB = [-15.0, -15.0, 0.0, -np.pi]
GRID_UB = [15.0, 15.0, 4.0, np.pi]
GRID_NSTEPS = [64, 64, 32, 32]

# Discretize the state space over which PDE is solved
# g = Grid(np.array([-4.0, -4.0, 0.0, -math.pi]), np.array([4.0, 4.0, 4.0, math.pi]), 4, np.array([60, 60, 20, 36]), [3])
g = Grid(np.array(GRID_LB), np.array(GRID_UB), 4, np.array(GRID_NSTEPS), [3])

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

# form HJ-Reach-based agent for Dubins4d Reach-Avoid environment
hjreach_agent = HJReachDubins4dReachAvoidAgent(grid=g, dynamics=my_car, goal=goal, obstacle=obstacle, time_grid=tau)

# get value function at last timestep
V_final = hjreach_agent.hji_values[..., 0]

# Compute spatial derivatives of final value function at every state
x_derivative = computeSpatDerivArray(g, V_final, deriv_dim=1, accuracy="low")
y_derivative = computeSpatDerivArray(g, V_final, deriv_dim=2, accuracy="low")
v_derivative = computeSpatDerivArray(g, V_final, deriv_dim=3, accuracy="low")
th_derivative = computeSpatDerivArray(g, V_final, deriv_dim=4, accuracy="low")

# specify arbitrary state
x0 = -0.5
y0 = -0.5
v0 = 1.0
th0 = math.pi/4
X0 = [x0,y0,v0,th0]

action = hjreach_agent.get_action(X0)
print("Active control: {}".format(action['active_ctrl']))
# print("Optimal accel is {}\n".format(opt_a))
# print("Optimal rotation is {}\n".format(opt_w))

print("DONE!")