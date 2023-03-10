# An example of running a random agent on the dubins4d_reachavoid environment

import time
import numpy as np
from dubins4d_reachavoid import Dubins4dReachAvoidEnv

# create environment
env = Dubins4dReachAvoidEnv(time_accel_factor=10.0, render_mode="human")

# get initial observation and info
obs, info = env.reset()

# take random actions with random delays until episode is done
while True:

    # random delay to allow system to propagate
    time.sleep(np.random.rand()/5)

    # random action for next time interval
    obs, rew, done, info = env.step_to_now(env.action_space.sample())

    if done:
        break

print("Done!\nReward = {}\nInfo: {}".format(rew, info))
