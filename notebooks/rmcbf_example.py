# Example of a risk-metric control barrier function
# for 1D Double integrator System

import numpy as np
import matplotlib.pyplot as plt
import itertools

from pyrmm.environments.dubins4d_reachavoid import cvx_qp_solver

###
# Problem parameters
###

# obstacle location on x-axis
p_obs = 5.0 # [m]

# control bounds
u1_min = -1.0   # [m/s/s]
u1_max = 1.0    # [m/s/s]

# evaluate derivatives at current state, i.e. local origin
xtil_1 = 0.0
xtil_2 = 0.0

# initial/current state of system, or 
# state at which observation z is taken
# p_z = 2.9   # [m] initial positon
# v_z = 2.0   # [m/s]
p_z_arr = np.linspace(-1.0, 4.5)
# v_z_arr = np.array([-1, -0.1, 0.1, 1.0, 2.0])
v_z_arr = np.arange(-1, 2.5, 0.5)

results = np.empty((len(v_z_arr), len(p_z_arr)))

# plotting params
marker = itertools.cycle((',', '+', '.', 'o', '*')) 

for i, v_z in enumerate(v_z_arr):
    for j, p_z in enumerate(p_z_arr):

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

        # break if stopping impossible
        if p_minstop > p_obs:
            results[i,j] = np.nan
            continue

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
        lambda_h = 0.1

        # Constraint: Left side of <= inequality
        G = np.reshape([-Lghx1], (1,1))

        # Constraint: right side of <= inequality
        h = np.reshape([Lfhx + lambda_h * h_x], (1,1))

        ### 
        # solve QP
        ###

        u_opt = cvx_qp_solver(P=P, q=q, G=G, h=h)
        results[i,j] = u_opt[0]

        # print("Min-norm safe control = ", u_opt[0])

    plt.plot(p_z_arr, results[i], marker = next(marker), label="v={} m/s".format(v_z))

plt.title("Safe Control Input As Function of Position and Velocity")
plt.xlabel("position [m]")
plt.ylabel("control input [m/s/s]")
plt.legend()
plt.show()