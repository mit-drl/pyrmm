# example script for solving quadratic program for dubins4d

from pyrmm.cbfs.dubins4d_reachavoid_agent import CBFDubins4dReachAvoidAgent
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv

# create environment
env = Dubins4dReachAvoidEnv(time_accel_factor=10, render_mode='human')

# speed constraints
vmin = 0    # [m/s]
vmax = 2    # [m/s]

# control constraints
u1min = -0.2    # [rad/s]
u1max = 0.2     # [rad/s]
u2min = -0.5    # [m/s/s]
u2max = 0.5     # [m/s/s]

# obstacle barrier function parameters
alpha_p1 = 0.7535
alpha_p2 = 0.6664
alpha_q1 = 1.0045
alpha_q2 = 1.0267

# decay rate bounds on speed barrier functions
gamma_vmax = 1
gamma_vmin = 1

# decay rate bounds on heading and speed lyapunov functions
lambda_Vtheta = 1
lambda_Vspeed = 1

# penalty values on slack variables in objective function
p_Vtheta = 1
p_Vspeed = 1

# create agent
agent = CBFDubins4dReachAvoidAgent(
    goal=env._goal,
    obstacle=env._obstacle,
    vmin=vmin,
    vmax=vmax,
    u1min=u1min,
    u1max=u1max,
    u2min=u2min,
    u2max=u2max,
    alpha_p1=alpha_p1,
    alpha_p2=alpha_p2,
    alpha_q1=alpha_q1,
    alpha_q2=alpha_q2,
    gamma_vmin=gamma_vmin,
    gamma_vmax=gamma_vmax,
    lambda_Vtheta=lambda_Vtheta,
    lambda_Vspeed=lambda_Vspeed,
    p_Vtheta=p_Vtheta,
    p_Vspeed=p_Vspeed
)

# run agent until termination
env.reset()
while True:

    # get agent at current state
    action = agent.get_action(state=env._Dubins4dReachAvoidEnv__state)

    # random action for next time interval
    obs, rew, done, info = env.step_to_now(action)

    if done:
        break

