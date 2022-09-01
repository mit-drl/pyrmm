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
from odp.solver import HJSolver, computeSpatDerivArray


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

# get value function at last timestep
V_final = result[..., 0]

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

# interpolate derivative at arbitrary state
spat_deriv_vector = (
    interpn(g.grid_points, x_derivative, X0), 
    interpn(g.grid_points, y_derivative, X0),
    interpn(g.grid_points, v_derivative, X0),
    interpn(g.grid_points, th_derivative, X0)
)

def dubins4d_optCtrl(dubins4d_car, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return:
        """
        # System dynamics
        # x_dot     = v * cos(theta) + d_1
        # y_dot     = v * sin(theta) + d_2
        # v_dot = a
        # theta_dot = w
        opt_a = dubins4d_car.uMax[0]
        opt_w = dubins4d_car.uMax[1]

        # The initialized control only change sign in the following cases
        if dubins4d_car.uMode == "min":
            if spat_deriv[2] > 0:
                opt_a = dubins4d_car.uMin[0]
            if spat_deriv[3] > 0:
                opt_w = dubins4d_car.uMin[1]
        else:
            if spat_deriv[2] < 0:
                opt_a = dubins4d_car.uMin[0]
            if spat_deriv[3] < 0:
                opt_w = dubins4d_car.uMin[1]

        return opt_a, opt_w

# Compute the optimal control
opt_a, opt_w = dubins4d_optCtrl(my_car, spat_deriv_vector)
print("Optimal accel is {}\n".format(opt_a))
print("Optimal rotation is {}\n".format(opt_w))

print("DONE!")