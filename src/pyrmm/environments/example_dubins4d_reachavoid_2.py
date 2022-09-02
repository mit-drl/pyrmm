# An example of running a random agent on the dubins4d_reachavoid environment

import time
import numpy as np
from dubins4d_reachavoid import Dubins4dReachAvoidEnv, CircleRegion

# create environment
env = Dubins4dReachAvoidEnv(time_accel_factor=10.0, render_mode="human")

# get initial observation and info
obs, info = env.reset()

# move obstacle far away to avoid collision
env._obstacle = CircleRegion(xc=-100, yc=-100, r=0.1)

# always inactive control to test if CLF drives to goal
action = env.action_space.sample()
action['active_ctrl'] = False

# take random actions with random delays until episode is done
while True:

    # random delay to allow system to propagate
    # time.sleep(np.random.rand())

    # random action for next time interval
    obs, rew, done, info = env.step_to_now(action)

    if done:
        break

print("Done!\nReward = {}\nInfo: {}".format(rew, info))
