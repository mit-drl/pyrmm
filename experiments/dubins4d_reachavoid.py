# Hydrazen app run benchmarking experiments for random, LRMM, CBF, and HJ-Reach algorithsm
# on the dubins4d_reachavoid environment

import hydra
import time
import multiprocess
import numpy as np

from functools import partial
from hydra.core.config_store import ConfigStore
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv
from hydra_zen import make_config, instantiate, make_custom_builds_fn, builds

_CONFIG_NAME = "dubins4d_reachavoid_experiment"
_MONITOR_RATE = 1

def execute_random_agent(env, max_delay:float=0.1):
    '''run random agent until episode completioin

    Args:
        env : Dubins4dReachAvoidEnv
            gym environment for agent interaction
        max_delay : float [s]
            maximum delay between steps
    '''

    # reset env to restart timing
    env.reset()

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
    'time_accel': DEFAULT_TIME_ACCEL,
    'n_cores': multiprocess.cpu_count() # number of cores for multiprocessing jobs
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

    # create pool of multiprocess jobs
    pool = multiprocess.Pool(obj.n_cores)

    # create storage for results
    results = dict()

    # create list of environment objects to be distributed during multiprocessing
    envs = [Dubins4dReachAvoidEnv(time_accel_factor=obj.time_accel) for i in range(obj.n_trials)]

    # create partial function for distributing envs to random agent executor
    part_execute_random_agent = partial(execute_random_agent)

    # use iterative map for process tracking
    t_start = time.time()
    randagent_iter = pool.imap(part_execute_random_agent, envs)

    # track multiprocess progress
    results['rand'] = []
    for i,_ in enumerate(envs):
        results['rand'].append(randagent_iter.next())
        if i%_MONITOR_RATE ==  0:
            print("Random-agent trial: completed {} of {} after {:.2f}".format(i+1, len(envs), time.time()-t_start,))

    pool.close()
    pool.join()

if __name__ == "__main__":
    task_function()