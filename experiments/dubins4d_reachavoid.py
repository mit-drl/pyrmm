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
from hydra_zen import make_config, instantiate, make_custom_builds_fn, builds
from typing import List, Dict
from numpy.typing import ArrayLike

import pyrmm.utils.utils as U

from odp.dynamics import DubinsCar4D as DubinsCar4DODP
from odp.Grid import Grid
from odp.Shapes import CylinderShape

from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv
from pyrmm.hjreach.dubins4d_reachavoid_agent import HJReachDubins4dReachAvoidAgent

_CONFIG_NAME = "dubins4d_reachavoid_experiment"
_MONITOR_RATE = 1
_SAVE_FNAME = U.format_save_filename(Path(__file__), 5)
_SMALL_NUMBER = 1e-5

# dictionary keys
K_RANDOM_AGENT = 'random_agent'
K_HJREACH_AGENT = 'hjreach_agent'
# K_N_TRIALS = 'n_trials'
# K_TIME_ACCEL = 'time_accel'
# K_N_CORES = 'n_cores'
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
    '''run random agent until episode completion

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

def execute_hjreach_agent(env,
    time_horizon : float,
    time_step : float,
    grid_lb : ArrayLike,
    grid_ub : ArrayLike,
    grid_nsteps : ArrayLike)->Dict:
    '''run HJ-Reachability agent until episode completion

    Args:
        time_horizon : float
            horizon to compute backward reachable set [s]
        time_step : float
            size of time steps to reach horizon
        grid_lb : ArrayLike[float]:
            lower bounds on each dimension in discretized state grid
        grid_ub : ArrayLike[float]:
            upper bounds on each dimension in discretized state grid
        grid_nsteps : ArrayLike[int]:
            number of discretization points each dimension of state grid
    '''

    # agent properties that can be instantiated a priori to environment
    dynamics = DubinsCar4DODP(uMode="min", dMode="max", dMin = [0.0,0.0], dMax = [0.0,0.0])
    time_grid = np.arange(start=0, stop=time_horizon + _SMALL_NUMBER, step=time_step)
    grid = Grid(
        minBounds=np.asarray(grid_lb), 
        maxBounds=np.asarray(grid_ub), 
        dims=4, 
        pts_each_dim=np.asarray(grid_nsteps), 
        periodicDims=[3])

    # reset env to restart timing and get obstacle and goal locations
    obs, info = env.reset()

    # extract and encode explicit obstacle and goal regions
    # NOTE: this access private information about the enviornment, giving HJ-reach
    # and advantage
    goal = CylinderShape(grid=grid, ignore_dims=[2,3], center=np.array([env._goal.xc, env._goal.yc, 0, 0]), radius=env._goal.r)
    obstacle = CylinderShape(grid=grid, ignore_dims=[2,3], center=np.array([env._obstacle.xc, env._obstacle.yc, 0, 0]), radius=env._obstacle.r)

    # instantiate the HJ-reach agent (which solves for HJI value function on grid)
    hjreach_agent = HJReachDubins4dReachAvoidAgent(grid=grid, dynamics=dynamics, goal=goal, obstacle=obstacle, time_grid=time_grid)

    while True:

        # get current state and transform to odp's x-y-v-theta ordering
        # NOTE: this access private information about the enviornment, giving HJ-reach
        # and advantage
        state = env._Dubins4dReachAvoidEnv__state
        state[2], state[3] = state[3], state[2]

        # compute action at current state
        action = hjreach_agent.get_action(state=state)

        # employ action in environment
        obs, rew, done, info = env.step_to_now(action)

        if done:
            break

    return info


# def env_agent_trial_runner(env_cfg, agent_runner_cfg, dummy_var):
#     '''instantiates environment and agent and runs single-episode trial'''

#     # instantiate environment
#     env = instantiate(env_cfg)

#     # execute agent with environment instance
#     agent_runner = instantiate(agent_runner_cfg)
#     info = agent_runner(env=env)

#     # close environment
#     env.close()

#     # return info
#     return info

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# Configure environment
DEFAULT_TIME_ACCEL = 10.0   # sim-tim acceleration factor
EnvConf = builds(Dubins4dReachAvoidEnv,
    time_accel_factor=DEFAULT_TIME_ACCEL
)

# Configure random agent
# e.g. configure random agent max_delay so it's recorded by hydra
DEFAULT_RANDOM_AGENT_MAX_DELAY = 0.1
RandomAgentConf = pbuilds(execute_random_agent, max_delay=DEFAULT_RANDOM_AGENT_MAX_DELAY)

# Configure HJ-reach agent
DEFAULT_HJREACH_TIME_HORIZON = 1.0
DEFAULT_HJREACH_TIME_STEP = 0.05
DEFAULT_HJREACH_GRID_LB = [-15.0, -15.0, 0.0, -np.pi]
DEFAULT_HJREACH_GRID_UB = [15.0, 15.0, 4.0, np.pi]
DEFAULT_HJREACH_GRID_NSTEPS = [64, 64, 32, 32]
HJReachConf = pbuilds(execute_hjreach_agent, 
    time_horizon = DEFAULT_HJREACH_TIME_HORIZON,
    time_step = DEFAULT_HJREACH_TIME_STEP,
    grid_lb = DEFAULT_HJREACH_GRID_LB,
    grid_ub = DEFAULT_HJREACH_GRID_UB,
    grid_nsteps = DEFAULT_HJREACH_GRID_NSTEPS)

# Top-level configuration of experiment
DEFAULT_N_TRIALS = 4       # number of trials (episodes) per agent
agent_config_inputs = {
    K_RANDOM_AGENT: RandomAgentConf,
    K_HJREACH_AGENT: HJReachConf
}
ExpConfig = make_config(
    n_trials = DEFAULT_N_TRIALS,
    n_cores = multiprocessing.cpu_count(),
    env = EnvConf,
    **agent_config_inputs
    # random_agent = RandomAgentConf
)

# store for command line interface
ConfigStore.instance().store(_CONFIG_NAME,ExpConfig)

##############################################
############### TASK FUNCTIONS ###############
##############################################
# a logger for this file
log = logging.getLogger(__name__)
@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: ExpConfig):

    # instantiate the experiment objects
    # obj = instantiate(cfg)

    # create storage for results
    results = dict()

    for agent_name in agent_config_inputs.keys():

        # create pool of multiprocess jobs
        pool = multiprocessing.Pool(cfg.n_cores)

        # create list of environment objects to be distributed during multiprocessing
        # envs = [Dubins4dReachAvoidEnv(time_accel_factor=cfg.time_accel) for i in range(cfg.n_trials)]
        envs = [instantiate(cfg.env) for _ in range(cfg.n_trials)]

        # create partial function for distributing envs to random agent executor
        p_agent_runner = partial(instantiate(getattr(cfg,agent_name)))

        # create partial function of env_agent_trial_runner
        # part_env_agent_trial_runner = partial(env_agent_trial_runner, cfg.env, cfg.random_agent)

        # create list of agent configs, one for each trial
        # agent_runner_cfgs = [cfg.random_agent for i in range(cfg.n_trials)]
        # dummy_iter = range(cfg.n_trials)

        # use iterative map for process tracking
        t_start = time.time()
        randagent_iter = pool.imap(p_agent_runner, envs)
        # randagent_iter = pool.imap(part_env_agent_trial_runner, dummy_iter)

        # track multiprocess progress
        results[agent_name] = {K_TRIAL_DATA:[]}
        for i in range(cfg.n_trials):
            results[agent_name][K_TRIAL_DATA].append(randagent_iter.next())
            if i%_MONITOR_RATE ==  0:
                print("{} trial: completed {} of {} after {:.2f}".format(agent_name, i+1, cfg.n_trials, time.time()-t_start,))

        pool.close()
        pool.join()

        # compute aggregate metrics
        results[agent_name][K_AGGREGATE_DATA] = aggregate_agent_metrics(results[agent_name][K_TRIAL_DATA])

        # log aggregate results
        log.info("Agent: {} trials complete with aggregated results:\n{}".format(agent_name, results[agent_name][K_AGGREGATE_DATA]))

    # save (pickle) results
    with open(_SAVE_FNAME+'.pkl', 'wb') as handle:
        pickle.dump(results, handle)



if __name__ == "__main__":
    task_function()