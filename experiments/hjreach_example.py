"""
Example that visualize behavior of HJ-Reach agent
"""

import numpy as np
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv, CircleRegion
from dubins4d_reachavoid import execute_hjreach_agent

env = Dubins4dReachAvoidEnv(
    time_accel_factor=10.0,
    render_mode="human",
    n_obstacles=1,
    obstacle_min_rad=4.5, 
    obstacle_max_rad=11.5)

TIME_HORIZON = 4.0
TIME_STEP = 0.1
GRID_LB = [-15.0, -15.0, 0.0, -np.pi]
GRID_UB = [15.0, 15.0, 4.0, np.pi]
GRID_NSTEPS = [64, 64, 32, 32]
ACC_MIN, ACC_MAX = [-0.5, 0.5]
OMG_MIN, OMG_MAX = [-0.2, 0.2]

info = execute_hjreach_agent(
    env_n_seed=(env,15),
    time_horizon = TIME_HORIZON,
    time_step = TIME_STEP,
    grid_lb = GRID_LB,
    grid_ub = GRID_UB,
    grid_nsteps = GRID_NSTEPS,
    acc_min = ACC_MIN,
    acc_max = ACC_MAX,
    omg_min = OMG_MIN,
    omg_max = OMG_MAX,
    precompute_time_reset = True)

print(info)