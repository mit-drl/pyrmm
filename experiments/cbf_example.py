"""
Example that visualize behavior of CBF agent
"""

from dubins4d_reachavoid import execute_cbf_agent
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv

env = Dubins4dReachAvoidEnv(time_accel_factor=10.0,render_mode="human")

# Configure CBF agent
DEFAULT_VMIN = 0    # [M/S]
DEFAULT_VMAX = 2    # [M/S]
DEFAULT_U1MIN = -0.2    # [RAD/S]
DEFAULT_U1MAX = 0.2     # [RAD/S]
DEFAULT_U2MIN = -0.5    # [M/S/S]
DEFAULT_U2MAX = 0.5     # [M/S/S]
DEFAULT_ALPHA_P1 = 0.7535
DEFAULT_ALPHA_P2 = 0.6664
DEFAULT_ALPHA_Q1 = int(1)
DEFAULT_ALPHA_Q2 = int(1)
DEFAULT_GAMMA_VMAX = 1
DEFAULT_GAMMA_VMIN = 1
DEFAULT_LAMBDA_VTHETA = 1
DEFAULT_LAMBDA_VSPEED = 1
DEFAULT_P_VTHETA = 1
DEFAULT_P_VSPEED = 1


info = execute_cbf_agent(
    env_n_seed=(env, 23),
    vmin = DEFAULT_VMIN,
    vmax = DEFAULT_VMAX,
    u1min = DEFAULT_U1MIN,
    u1max = DEFAULT_U1MAX,
    u2min = DEFAULT_U2MIN,
    u2max = DEFAULT_U2MAX,
    alpha_p1 = DEFAULT_ALPHA_P1,
    alpha_p2 = DEFAULT_ALPHA_P2,
    alpha_q1 = DEFAULT_ALPHA_Q1,
    alpha_q2 = DEFAULT_ALPHA_Q2,
    gamma_vmin = DEFAULT_GAMMA_VMIN,
    gamma_vmax = DEFAULT_GAMMA_VMAX,
    lambda_Vtheta = DEFAULT_LAMBDA_VTHETA,
    lambda_Vspeed = DEFAULT_LAMBDA_VSPEED,
    p_Vtheta = DEFAULT_P_VTHETA,
    p_Vspeed = DEFAULT_P_VSPEED,
)

print(info)