# Example of a risk-metric control barrier function
# for 1D Double integrator System

import numpy as np

from pyrmm.environments.dubins4d_reachavoid import cvx_qp_solver

###
# Problem parameters
###

# obstacle location on x-axis
p_obs = 5.0 # [m]

# initial/current state of system, or 
# state at which observation z is taken
p_z = 2.9   # [m] initial positon
v_z = 2.0   # [m/s]

# control bounds
u1_min = -1.0   # [m/s/s]
u1_max = 1.0    # [m/s/s]

# evaluate derivatives at current state, i.e. local origin
xtil_1 = 0.0
xtil_2 = 0.0

###
# Computed parameters
###

# control input for minimum stopping distance
if v_z >= 0:
    u1_minstop = u1_min
else:
    u1_minstop = u1_max

# compute final position of minimum-distance stopping
p_minstop = -0.5 * v_z**2 / u1_minstop + p_z

# risk metric at current state
rho_x = np.exp(-(p_obs - p_minstop))

# barrier function value at current state
h_x = 1 - rho_x

###
# Compute Lie derivatives
###


# Lie derivative along g(x)
Lghx1 = ((xtil_2 + v_z) / u1_minstop) * np.exp(p_minstop - p_obs)

# Lie derivative along f(x)
Lfhx = -(xtil_2 + v_z) * np.exp(p_minstop - p_obs)

### 
# Formulate QP
###

# objective: minimize control input
P = np.eye(1).reshape(1,1)
q = np.zeros(1).reshape(1,1)
lambda_h = 1.0

# Constraint: Left side of <= inequality
G = np.reshape([-Lghx1], (1,1))

# Constraint: right side of <= inequality
h = np.reshape([Lfhx + lambda_h * h_x], (1,1))

### 
# solve QP
###

u_opt = cvx_qp_solver(P=P, q=q, G=G, h=h)

print("Min-norm safe control = ", u_opt)

