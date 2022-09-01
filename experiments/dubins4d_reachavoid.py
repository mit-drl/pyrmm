# Hydrazen app run benchmarking experiments for random, LRMM, CBF, and HJ-Reach algorithsm
# on the dubins4d_reachavoid environment

import hydra
import time
import multiprocessing
import logging
import pickle
import numpy as np

from pathlib import Path
from functools import partial
from hydra.core.config_store import ConfigStore
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv
from hydra_zen import make_config, instantiate, make_custom_builds_fn, builds
from typing import List, Dict

import pyrmm.utils.utils as U

_CONFIG_NAME = "dubins4d_reachavoid_experiment"
_MONITOR_RATE = 1
_SAVE_FNAME = U.format_save_filename(Path(__file__), 5)

# dictionary keys
K_RANDOM_AGENT = 'random_agent'
K_N_TRIALS = 'n_trials'
K_TIME_ACCEL = 'time_accel'
K_N_CORES = 'n_cores'
K_TRIAL_DATA = 'trial_data'
K_AGGREGATE_DATA = 'aggregate_data'
K_AVG_POLICY_COMPUTE_WALL_TIME = 'avg_policy_compute_wall_time'
K_AVG_ACTIVE_CTRL_STEPS = 'avg_active_ctrl_steps'
K_AVG_ACTIVE_CTRL_SIM_TIME = 'avg_active_ctrl_sim_time'
K_GOAL_COMPLETION_RATE = 'goal_completion_rate'
K_OBST_COLLISION_RATE = 'obst_collision_rate'
K_TIMEOUT_RATE = 'timeout_rate'
K_AVG_SIM_TIME_TO_GOAL = 'avg_sim_time_to_goal'

def aggregate_agent_metrics(trial_data:List)->Dict:
    '''compute metrics for agent performance for sequence of trials (episodes)
    Returns:
        AVG_POLICY_COMPUTE_WALL_TIME : average wall-clock computation time of agent's policy per step
        AVG_ACTIVE_CTRL_STEPS : average number of active control steps per episode
        AVG_ACTIVE_CTRL_SIM_TIME: average sim time spent under active control per episode
        GOAL_COMPLETION_RATE : fraction of episodes that end in the goal terminal state
        OBST_COLLISION_RATE: fraction of episodes that end in an obstacle collision
        TIMEOUT_RATE: fraction of episodes that end in env timeout
        AVG_SIM_TIME_TO_GOAL: average sim time to reach goal per episode with goal completion
    '''
    agg_data = dict()

    agg_data[K_AVG_POLICY_COMPUTE_WALL_TIME] = np.sum([t['cum_wall_clock_time'] for t in trial_data]) / np.sum([t['n_env_steps'] for t in trial_data])
    agg_data[K_AVG_ACTIVE_CTRL_STEPS] = np.mean([t['n_active_ctrl_env_steps'] for t in trial_data])
    agg_data[K_AVG_ACTIVE_CTRL_SIM_TIME] = np.mean([t['active_ctrl_sim_time'] for t in trial_data]) 
    agg_data[K_GOAL_COMPLETION_RATE] = len([t for t in trial_data if np.isclose(t['cum_reward'],1.0)])/len(trial_data)
    agg_data[K_OBST_COLLISION_RATE] = len([t for t in trial_data if np.isclose(t['cum_reward'],-1.0)])/len(trial_data)
    agg_data[K_TIMEOUT_RATE] = len([t for t in trial_data if np.isclose(t['cum_reward'],0.0)])/len(trial_data)
    agg_data[K_AVG_SIM_TIME_TO_GOAL] = np.mean([t['cum_sim_time'] for t in trial_data if np.isclose(t['cum_reward'],1.0)])

    return agg_data


def execute_random_agent(env, max_delay:float=0.1)->Dict:
    '''run random agent until episode completioin

    Args:
        env : Dubins4dReachAvoidEnv
            gym environment for agent interaction
        max_delay : float [s]
            maximum delay between steps

    Returns 
        info : Dict
            dictionary of episode metric info
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

# Configure random agent
# e.g. configure random agent max_delay so it's recorded by hydra
# TODO

# Top-level configuration and store for command line interface
make_config_input = {
    K_N_TRIALS: DEFAULT_N_TRIALS,   
    K_TIME_ACCEL: DEFAULT_TIME_ACCEL,
    K_N_CORES: multiprocessing.cpu_count() # number of cores for multiprocessing jobs
}
Config = make_config(**make_config_input)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################
# a logger for this file
log = logging.getLogger(__name__)
@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config):

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # create pool of multiprocess jobs
    pool = multiprocessing.Pool(obj.n_cores)

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
    results[K_RANDOM_AGENT] = {K_TRIAL_DATA:[]}
    for i,_ in enumerate(envs):
        results[K_RANDOM_AGENT][K_TRIAL_DATA].append(randagent_iter.next())
        if i%_MONITOR_RATE ==  0:
            print("Random-agent trial: completed {} of {} after {:.2f}".format(i+1, len(envs), time.time()-t_start,))

    pool.close()
    pool.join()

    # compute aggregate metrics
    results[K_RANDOM_AGENT][K_AGGREGATE_DATA] = aggregate_agent_metrics(results[K_RANDOM_AGENT][K_TRIAL_DATA])

    # log and save (pickle) results
    log.info("Agent: {} trials complete with aggregated results:\n{}".format(K_RANDOM_AGENT, results[K_RANDOM_AGENT][K_AGGREGATE_DATA]))
    with open(_SAVE_FNAME+'.pkl', 'wb') as handle:
        pickle.dump(results, handle)



if __name__ == "__main__":
    task_function()