# Hydrazen app run benchmarking experiments for random, LRMM, CBF, and HJ-Reach algorithsm
# on the dubins4d_reachavoid environment

import hydra
import time
import numpy as np

from hydra.core.config_store import ConfigStore
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv
from hydra_zen import make_config, instantiate, make_custom_builds_fn, builds

_CONFIG_NAME = "dubins4d_reachavoid_experiment"

def execute_random_agent(env, max_delay:float=0.1):
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

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

DEFAULT_N_TRIALS = 10       # number of trials (episodes) per agent
DEFAULT_TIME_ACCEL = 10.0   # sim-tim acceleration factor

# Top-level configuration and store for command line interface
make_config_input = {
    'n_trials': DEFAULT_N_TRIALS,
    'time_accel': DEFAULT_TIME_ACCEL
}
Config = make_config(**make_config_input)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config):

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # create storage for results
    results = dict()
    results['rand'] = []

    for i in range(obj.n_trials):
        # create environment
        env = Dubins4dReachAvoidEnv(time_accel_factor=obj.time_accel)

        # run random agent
        info_randagent = execute_random_agent(env)

        # save data
        # TODO: this is a placeholder print function
        print("random-agent: {}".format(info_randagent))
        results['rand'].append(info_randagent)

if __name__ == "__main__":
    task_function()