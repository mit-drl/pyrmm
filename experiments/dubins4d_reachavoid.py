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

from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv, K_ACTIVE_CTRL
from pyrmm.hjreach.dubins4d_reachavoid_agent import HJReachDubins4dReachAvoidAgent
from pyrmm.agents.dubins4d_reachavoid_agent import LRMMDubins4dReachAvoidAgent
from pyrmm.cbfs.dubins4d_reachavoid_agent import CBFDubins4dReachAvoidAgent

_CONFIG_NAME = "dubins4d_reachavoid_experiment"
_MONITOR_RATE = 1
_SAVE_FNAME = U.format_save_filename(Path(__file__), 5)
_SMALL_NUMBER = 1e-5

# dictionary keys
K_INACTIVE_AGENT = 'inactive_agent'
K_RANDOM_AGENT = 'random_agent'
K_HJREACH_AGENT = 'hjreach_agent'
K_LRMM_AGENT = 'lrmm_agent'
K_CBF_AGENT = 'cbf_agent'
# K_N_TRIALS = 'n_trials'
# K_TIME_ACCEL = 'time_accel'
# K_N_CORES = 'n_cores'
K_TRIAL_DATA = 'trial_data'
K_AGGREGATE_DATA = 'aggregate_data'
K_ERROR_RATE = 'error_rate'
K_AVG_POLICY_COMPUTE_WALL_TIME = 'avg_policy_compute_wall_time'
K_AVG_WALL_CLOCK_TIME_PER_EPISODE = 'avg_wall_clock_time_per_episode'
K_AVG_STEPS_PER_EPISODE = 'avg_steps_per_episode'
K_AVG_SIM_TIME_PER_EPISODE = 'avg_sim_time_per_episode'
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
        AVG_WALL_CLOCK_TIME_PER_EPISODE : average wall clock time per episode
        AVG_STEPS_PER_EPISODE : average number of control steps, active or inactive, per episode
        AVG_SIM_TIME_PER_EPISODE : average simulation time per episode
        AVG_ACTIVE_CTRL_STEPS : average number of active control steps per episode
        AVG_ACTIVE_CTRL_SIM_TIME: average sim time spent under active control per episode
        GOAL_COMPLETION_RATE : fraction of episodes that end in the goal terminal state
        OBST_COLLISION_RATE: fraction of episodes that end in an obstacle collision
        TIMEOUT_RATE: fraction of episodes that end in env timeout
        AVG_SIM_TIME_TO_GOAL: average sim time to reach goal per episode with goal completion
    '''
    agg_data = dict()

    agg_data[K_ERROR_RATE] = len([t for t in trial_data if t is None])/len(trial_data)

    # non-errored data
    dat = [t for t in trial_data if t is not None]
    agg_data[K_AVG_POLICY_COMPUTE_WALL_TIME] = np.sum([t['cum_wall_clock_time'] for t in dat]) / np.sum([t['n_env_steps'] for t in dat])
    agg_data[K_AVG_WALL_CLOCK_TIME_PER_EPISODE] = np.mean([t['cum_wall_clock_time'] for t in dat])
    agg_data[K_AVG_STEPS_PER_EPISODE] = np.mean([t['n_env_steps'] for t in dat])
    agg_data[K_AVG_SIM_TIME_PER_EPISODE] = np.mean([t['cum_sim_time'] for t in dat])
    agg_data[K_AVG_ACTIVE_CTRL_STEPS] = np.mean([t['n_active_ctrl_env_steps'] for t in dat])
    agg_data[K_AVG_ACTIVE_CTRL_SIM_TIME] = np.mean([t['active_ctrl_sim_time'] for t in dat]) 
    agg_data[K_GOAL_COMPLETION_RATE] = len([t for t in dat if np.isclose(t['cum_reward'],1.0)])/len(dat)
    agg_data[K_OBST_COLLISION_RATE] = len([t for t in dat if np.isclose(t['cum_reward'],-1.0)])/len(dat)
    agg_data[K_TIMEOUT_RATE] = len([t for t in dat if np.isclose(t['cum_reward'],0.0)])/len(dat)
    agg_data[K_AVG_SIM_TIME_TO_GOAL] = np.mean([t['cum_sim_time'] for t in dat if np.isclose(t['cum_reward'],1.0)])

    return agg_data

def execute_inactive_agent(env, delay:float)->Dict:
    '''run agent than never takes active control (env CLF always controls)

    Args:
        env : Dubins4dReachAvoidEnv
            gym environment for agent interaction
        delay : float [s]
            fixed delay between steps

    Returns 
        info : Dict
            dictionary of episode metric info
    '''
    assert delay > 0
    
    # set constant inactive action
    action = env.action_space.sample()
    action['active_ctrl'] = False

    # reset env to restart timing
    env.reset()

    while True:
        # random delay to allow system to propagate
        time.sleep(delay)

        # random action for next time interval
        obs, rew, done, info = env.step_to_now(action)

        if done:
            break

    return info

def execute_random_agent(env, max_delay:float)->Dict:
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

def execute_lrmm_agent(env,
    chkpt_file, active_ctrl_risk_threshold):
    
    # create LRMM agent from checkpointed model
    lrmm_agent = LRMMDubins4dReachAvoidAgent(
        chkpt_file=chkpt_file, 
        active_ctrl_risk_threshold=active_ctrl_risk_threshold
    )

    # reset env to restart timing
    obs, info = env.reset()

    while True:

        # compute action at current state
        action, ctrl_dur = lrmm_agent.get_action(observation=obs)

        # employ action in environment
        obs, rew, done, info = env.step_to_now(action)

        # wait part of ctrl duration (but enforce non-negative time)
        # time.sleep(max(0.1*ctrl_dur, 0.0))

        if done:
            break

    return info

def execute_cbf_agent(env,
    vmin, vmax,
    u1min, u1max,
    u2min, u2max,
    alpha_p1, alpha_p2, alpha_q1, alpha_q2,
    gamma_vmin, gamma_vmax,
    lambda_Vtheta, lambda_Vspeed,
    p_Vtheta, p_Vspeed):
    '''
    Args:
        vmin, vmax  : float
            min and max constraint on speed state variable
        u1min, u1max : float
            min and max constraint on turning rate control variable
        u2min, u2max : float
            min and max constraint on linear acceleration control variable
        alpha_p1, alpha_p2 : float
            penalty value for 2nd order parameterized method of HOCBF
        alpha_q1, alpha_q2 : float
            powers of 2nd order parameterized method of HOCBF
        gamma_vmax, gamma_vmin : float
            parameter lower-bounding evolution of speed barrier function
        lambda_Vtheta, lambda_Vspeed : float
            parameter upper-bounding evolution of headding and speed lyapunov functions
        p_Vtheta, p_Vspeed : float
            penalty in objective function on heading and speed slack variables that relax stability constraints
    '''

    # reset env to restart timing and get obstacle and goal locations
    obs, info = env.reset()

    # instantiate CBF agent
    # NOTE: this access private information about the enviornment, giving HJ-reach
    # and advantage
    cbf_agent = CBFDubins4dReachAvoidAgent(
        goal=env._goal, obstacle=env._obstacle,
        vmin = vmin,
        vmax = vmax,
        u1min = u1min,
        u1max = u1max,
        u2min = u2min,
        u2max = u2max,
        alpha_p1 = alpha_p1,
        alpha_p2 = alpha_p2,
        alpha_q1 = alpha_q1,
        alpha_q2 = alpha_q2,
        gamma_vmin = gamma_vmin,
        gamma_vmax = gamma_vmax,
        lambda_Vtheta = lambda_Vtheta,
        lambda_Vspeed = lambda_Vspeed,
        p_Vtheta = p_Vtheta,
        p_Vspeed = p_Vspeed
    )

    while True:

        # get current state
        # NOTE: this access private information about the enviornment, giving HJ-reach
        # and advantage
        state = env._Dubins4dReachAvoidEnv__state

        # try to compute action at current state,
        # catch value error corresponding to infeasible QP
        try:
            action = cbf_agent.get_action(state=state)
        except ValueError as e:
            if str(e) == "domain error":
                # infeasible QP, apply inactive control
                action = env.action_space.sample()
                action[K_ACTIVE_CTRL] = False
            else:
                raise


        # employ action in environment
        obs, rew, done, info = env.step_to_now(action)

        if done:
            break

    return info

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# Configure environment
DEFAULT_TIME_ACCEL = 10.0   # sim-tim acceleration factor
EnvConf = builds(Dubins4dReachAvoidEnv,
    time_accel_factor=DEFAULT_TIME_ACCEL
)

# Configure inactive agent
DEFAULT_INACTIVE_AGENT_DELAY = 0.05
InactiveAgentConf = pbuilds(execute_inactive_agent, delay=DEFAULT_INACTIVE_AGENT_DELAY)

# Configure random agent
# e.g. configure random agent max_delay so it's recorded by hydra
DEFAULT_RANDOM_AGENT_MAX_DELAY = 0.1
RandomAgentConf = pbuilds(execute_random_agent, max_delay=DEFAULT_RANDOM_AGENT_MAX_DELAY)

# Configure HJ-reach agent
DEFAULT_HJREACH_TIME_HORIZON = 1.0
DEFAULT_HJREACH_TIME_STEP = 0.1
DEFAULT_HJREACH_GRID_LB = [-15.0, -15.0, 0.0, -np.pi]
DEFAULT_HJREACH_GRID_UB = [15.0, 15.0, 4.0, np.pi]
DEFAULT_HJREACH_GRID_NSTEPS = [64, 64, 32, 32]
# DEFAULT_HJREACH_GRID_NSTEPS = [32, 32, 16, 16]
HJReachConf = pbuilds(execute_hjreach_agent, 
    time_horizon = DEFAULT_HJREACH_TIME_HORIZON,
    time_step = DEFAULT_HJREACH_TIME_STEP,
    grid_lb = DEFAULT_HJREACH_GRID_LB,
    grid_ub = DEFAULT_HJREACH_GRID_UB,
    grid_nsteps = DEFAULT_HJREACH_GRID_NSTEPS)

DEFAULT_LRMM_CHKPT_FILE = (
    "/home/ross/Projects/AIIA/risk_metric_maps/" +
    "outputs/2022-09-11/22-10-44/lightning_logs/" +
    "version_0/checkpoints/epoch=2027-step=442103.ckpt"
)
DEFAULT_LRMM_ACITVE_CTRL_RISK_THRESHOLD = 0.8
LRMMAgentConf = pbuilds(execute_lrmm_agent,
    chkpt_file = DEFAULT_LRMM_CHKPT_FILE,
    active_ctrl_risk_threshold = DEFAULT_LRMM_ACITVE_CTRL_RISK_THRESHOLD)

# Configure CBF agent
DEFAULT_VMIN = 0    # [M/S]
DEFAULT_VMAX = 2    # [M/S]
DEFAULT_U1MIN = -0.2    # [RAD/S]
DEFAULT_U1MAX = 0.2     # [RAD/S]
DEFAULT_U2MIN = -0.5    # [M/S/S]
DEFAULT_U2MAX = 0.5     # [M/S/S]
DEFAULT_ALPHA_P1 = 0.7535
DEFAULT_ALPHA_P2 = 0.6664
DEFAULT_ALPHA_Q1 = 1.0045
DEFAULT_ALPHA_Q2 = 1.0267
DEFAULT_GAMMA_VMAX = 1
DEFAULT_GAMMA_VMIN = 1
DEFAULT_LAMBDA_VTHETA = 1
DEFAULT_LAMBDA_VSPEED = 1
DEFAULT_P_VTHETA = 1
DEFAULT_P_VSPEED = 1
CBFAgentConf = pbuilds(execute_cbf_agent,
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
    p_Vspeed = DEFAULT_P_VSPEED
    )

# Top-level configuration of experiment
DEFAULT_N_TRIALS = 256       # number of trials (episodes) per agent
agent_config_inputs = {
    K_INACTIVE_AGENT: InactiveAgentConf,
    K_RANDOM_AGENT: RandomAgentConf,
    K_HJREACH_AGENT: HJReachConf,
    K_LRMM_AGENT: LRMMAgentConf,
    K_CBF_AGENT: CBFAgentConf
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

        # use iterative map for process tracking
        t_start = time.time()
        randagent_iter = pool.imap(p_agent_runner, envs)

        # track multiprocess progress
        results[agent_name] = {K_TRIAL_DATA:[]}
        for i in range(cfg.n_trials):
            try: 
                results[agent_name][K_TRIAL_DATA].append(randagent_iter.next())
            except ValueError:
                results[agent_name][K_TRIAL_DATA].append(None)

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