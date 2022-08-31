# Hydrazen app run benchmarking experiments for random, LRMM, CBF, and HJ-Reach algorithsm
# on the dubins4d_reachavoid environment

import time
import numpy as np
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv

def execute_random_agent(env, max_delay:float=0.01):
    '''run random agent until episode completioin

    Args:
        env : Dubins4dReachAvoidEnv
            gym environment for agent interaction
        max_delay : float [s]
            maximum delay between steps
    '''
    while True:
        # random delay to allow system to propagate
        time.sleep(np.random.uniform(0,max_delay))

        # random action for next time interval
        obs, rew, done, info = env.step_to_now(env.action_space.sample())

        if done:
            break

    return info


def task_function():

    # create environment
    env = Dubins4dReachAvoidEnv(time_accel_factor=10.0)

    # run random agent
    info_randagent = execute_random_agent(env)

    # save data
    # TODO: this is a placeholder print function
    print("random-agent: {}".format(info_randagent))

if __name__ == "__main__":
    task_function()