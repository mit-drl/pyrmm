'''
An environment for testing safety-based navigation algorithms on Dubins-like system dynamics.

Using OpenAI Gym API---not because this necessarily a reinforcement learning problem---
but because it offers a convienent, standardized interface for all algorithms
'''

import gym
import time
import numpy as np

from scipy.integrate import odeint
from cvxopt import solvers, matrix
from numpy.random import normal, uniform
from typing import Tuple, Optional
from numpy.typing import ArrayLike
from shapely.geometry import Point, LineString
from types import SimpleNamespace
from copy import deepcopy
from gym.utils.renderer import Renderer

# state space (SS) constants (e.g. vector indexes, bounds)
SS_XIND = 0     # index in state vector for x-positon
SS_YIND = 1     # # index in state vector for y-positon
SS_THETAIND = 2 # index in state vector for theta (heading)
SS_VIND = 3     # index in state vector for linear speed
SS_XMIN = -10.0     # [m]
SS_XMAX = 10.0      # [m]
SS_YMIN = -10.0     # [m]
SS_YMAX = 10.0      # [m]
SS_THETAMIN = -np.inf   # [rad]
SS_THETAMAX = np.inf    # [rad]
SS_VMIN = 0.0       # [m/s]
SS_VMAX = 2.0       # [m/s]

# control space (CS) parameters
K_ACTIVE_CTRL = "active_ctrl"
K_TURNRATE_CTRL = "turnrate_ctrl"
K_ACCEL_CTRL = "accel_ctrl"
# DURATION_CTRL = "duration_ctrl"
CS_DTHETAMIN = -0.25 # [rad/s]    
CS_DTHETAMAX = 0.25  # [rad/s]
CS_DVMIN = -0.5     # [m/s/s]
CS_DVMAX = 0.5      # [m/s/s]

# goal and obstacle params
GOAL_R_MIN = 0.1
GOAL_R_MAX = 1.0
OBST_R_MIN = 2.0
OBST_R_MAX = 5.0

# observation space (OS) parameters
OS_N_RAYS_DEFAULT = 12      # number of rays to cast
OS_RAY_MAX_DEFAULT = 5.0    # [m] maximum length of ray

# number of time steps to analyze per system propagation
PROPAGATE_TIMESTEPS = 16
MAX_EPISODE_SIM_TIME_DEFAULT = 100.0    # [s] simulated time
TIME_ACCEL_FACTOR_DEFAULT = 1.0         # [s-sim-time/s-wall-clock-time] acceleration of simulation time

# Control lyapunov QP parameters
GAMMA_VMIN_DEFAULT=1    # parameter lower-bounding evolution of v-min speed barrier function
GAMMA_VMAX_DEFAULT=1    # parameter lower-bounding evolution of v-max speed barrier function
LAMBDA_VTHETA_DEFAULT=1     # parameter upper-bounding evolution of headding lyapunov functions
LAMBDA_VSPEED_DEFAULT=1     # parameter upper-bounding evolution of headding lyapunov functions
P_VTHETA_DEFAULT=1      # penalty in objective function on heading slack variables that relax stability constraints
P_VSPEED_DEFAULT=1      # penalty in objective function on speed slack variables that relax stability constraints

# info dictionary keys
K_N_ENV_STEPS = 'n_env_steps'
K_N_ACTIVE_CTRL_ENV_STEPS = 'n_active_ctrl_env_steps'
K_CUM_ACTIVE_CTRL_SIM_TIME = 'active_ctrl_sim_time'
K_CUM_WALL_CLOCK_TIME = 'cum_wall_clock_time'
K_CUM_SIM_TIME = 'cum_sim_time'
K_AVG_POLICY_COMPUTE_TIME = 'avg_policy_compute_time'
K_CUM_REWARD = 'cum_reward'

class Dubins4dReachAvoidEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array", "single_rgb_array"], "render_fps": 15}

    # define action space as class attribute so that it is accessable by agents
    # without having to instantiate a "dummy env"
    action_space = gym.spaces.Dict({
        K_ACTIVE_CTRL: gym.spaces.Discrete(2),
        K_TURNRATE_CTRL: gym.spaces.Box(low=CS_DTHETAMIN, high=CS_DTHETAMAX),    # [rad/s]
        K_ACCEL_CTRL: gym.spaces.Box(low=CS_DVMIN, high=CS_DVMAX),         # [m/s/s]
    })

    def __init__(self, 
        n_rays: int = OS_N_RAYS_DEFAULT, 
        ray_length: float = OS_RAY_MAX_DEFAULT,
        max_episode_sim_time: float = MAX_EPISODE_SIM_TIME_DEFAULT,
        time_accel_factor: float = TIME_ACCEL_FACTOR_DEFAULT,
        gamma_vmin=GAMMA_VMIN_DEFAULT, 
        gamma_vmax=GAMMA_VMAX_DEFAULT,
        lambda_Vtheta=LAMBDA_VTHETA_DEFAULT, 
        lambda_Vspeed=LAMBDA_VSPEED_DEFAULT,
        p_Vtheta=P_VTHETA_DEFAULT, 
        p_Vspeed=P_VSPEED_DEFAULT,
        render_mode: Optional[str] = None):
        '''
        Args
            n_rays : int
                number of rays to cast for observations, evenly spaced
            ray_length : float
                maximum extent of ray if no collision with obstacle
            max_episode_sim_time : float
                elapsed sim time before timeout termination critera
            time_accel_factor : float
                sim time acceleration relative to wall clock time
            gamma_vmax, gamma_vmin : float
                parameter lower-bounding evolution of speed barrier function
            lambda_Vtheta, lambda_Vspeed : float
                parameter upper-bounding evolution of headding and speed lyapunov functions
            p_Vtheta, p_Vspeed : float
                penalty in objective function on heading and speed slack variables that relax stability constraints
        '''

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # Define the attribute render_mode in your environment

        # Timing parameters
        assert max_episode_sim_time > 0
        self._max_episode_sim_time = max_episode_sim_time   # [s] sim time until termination of episode

        assert time_accel_factor >= 1.0
        self._time_accel_factor = time_accel_factor         # [s-sim-time/s-wall-time] sim-time accleration factor relative to wall clock

        # define state space bounds
        # [x [m], y [m], theta [rad], v [m/s]]
        self.state_space = gym.spaces.Box(
            low = np.array((SS_XMIN, SS_YMIN, SS_THETAMIN, SS_VMIN)),
            high =  np.array((SS_XMAX, SS_YMAX, SS_THETAMAX, SS_VMAX))
        )

        # define observation space (state space distinct but related to observation space)
        # sim-time [s]
        # x-rel-to-goal [m]
        # y-rel-to-goal [m]
        # heading [rad]
        # speed [m/s]
        # ray-casts to obstacles [m]
        self._n_rays = n_rays
        self._max_ray_length = ray_length
        self.observation_space = gym.spaces.Box(
            low = np.concatenate(([0], 2*[-np.inf], [SS_THETAMIN, SS_VMIN], np.zeros(self._n_rays))),
            high = np.concatenate((3*[np.inf], [SS_THETAMAX, SS_VMAX], self._max_ray_length*np.ones(self._n_rays))),
        )

        # control and observaiton disturbances 
        # (private because solution algorithms should not know them)
        self.__dist = SimpleNamespace()
        self.__dist.ctrl = SimpleNamespace()
        self.__dist.ctrl.x_mean = 0.0
        self.__dist.ctrl.x_std = 0.01
        self.__dist.ctrl.y_mean = 0.0
        self.__dist.ctrl.y_std = 0.01
        self.__dist.ctrl.theta_mean = 0.0
        self.__dist.ctrl.theta_std = 0.001
        self.__dist.ctrl.v_mean = 0.0
        self.__dist.ctrl.v_std = 0.01

        # control lyapunov function params for inactive control
        self.gamma_vmin = gamma_vmin
        self.gamma_vmax = gamma_vmax
        self.lambda_Vtheta = lambda_Vtheta
        self.lambda_Vspeed = lambda_Vspeed
        self.p_Vtheta = p_Vtheta
        self.p_Vspeed = p_Vspeed

        # setup renderer
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode.
        """
        self.render_window_size = 512  # The size of the PyGame window
        if self.render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render

            pygame.init()
            pygame.display.init()
            self.render_window = pygame.display.set_mode((self.render_window_size, self.render_window_size))
            self.render_clock = pygame.time.Clock()
                
        # The following line uses the util class Renderer to gather a collection of frames 
        # using a method that computes a single frame. We will define _render_frame below.
        self.renderer = Renderer(self.render_mode, self._render_frame)

        # reset env to generate all instance attributes
        self.reset()

    def reset(self, seed=None):
        # seed random number generator self.np_random
        super().reset(seed=seed)

        # randomize initial state (x [m], y [m], theta [rad], v [m/s])
        # (private because solution algs should not know this explicitly)
        self.__state = self.state_space.sample()

        # specify zero action for environment init
        self._cur_action = self.action_space.sample()
        self._cur_action[K_ACTIVE_CTRL] = 0
        self._cur_action[K_TURNRATE_CTRL][0] = 0.0
        self._cur_action[K_ACCEL_CTRL][0] = 0.0

        # randomize goal, obstacle
        goal_xc, goal_yc = self.state_space.sample()[:2]
        goal_r = uniform(GOAL_R_MIN, GOAL_R_MAX)
        self._goal = CircleRegion(xc=goal_xc, yc=goal_yc, r=goal_r)

        # randomize obstacle (not meant for direct access)
        obst_xc, obst_yc = self.state_space.sample()[:2]
        obst_r = uniform(OBST_R_MIN, OBST_R_MAX)
        self._obstacle = CircleRegion(xc=obst_xc, yc=obst_yc, r=obst_r)

        # clean the render collection and add the initial frame
        self.renderer.reset()
        self.renderer.render_step()

        # reset sim clock and sim-to-wall clock sync point
        self._wall_clock_elapsed_time = 0.0
        self._sim_time = 0.0
        self._wall_clock_sync_time = time.time()

        # reset timing and stepping metrics
        self._n_env_steps = 0
        self._n_active_ctrl_env_steps = 0
        self._active_ctrl_sim_time = 0.0

        # return initial observation and information
        self._cum_reward = 0
        observation = self._get_observation()
        info = self._get_info(False)
        return observation, info

    def step(self, action):
        '''Unused in real-time paradigm
        '''
        raise NotImplementedError('Alternative paradigm for real-time environment. See step_to_now function ')
    
    def step_to_now(self, next_action):
        '''Advance system to current time, take observation, and specify action for next time interval

        Note: this is an alternative paradigm for stepping a decision process. 
        In standard paradigm, you apply an action at time t, propagate the system to time t+dt, 
        and return the observation at time t+dt.

        In this paradigm, you propagate the system from time t-dt to time t based on action
        set at time t-dt, return the observation at time t, and then specify 
        the action that will be applied from t to t+dt

        Args
            next_action : gym.spaces.Dict
                action to be applied during subsequent time step
                active_ctrl: boolean True if agent is taking active control of vehicle
                turnrate_ctrl: float rate of change of vehcile heading [rad/s]
                accel_ctrl: float linear acceleration of vehicle [m/s/s]
                ## duration_ctrl: float length of time to apply control
        '''

        self._n_env_steps += 1

        if not self.action_space.contains(self._cur_action):
            raise ValueError("Action {} outside of action space {}".format(self._cur_action, self.action_space))

        # format control portion of action if active control (otherwise passive CLF will control)
        ctrl = None
        if self._cur_action[K_ACTIVE_CTRL]:
            self._n_active_ctrl_env_steps += 1
            ctrl = np.concatenate((self._cur_action[K_TURNRATE_CTRL], self._cur_action[K_ACCEL_CTRL]))

        # propagate system from t-dt to t based on action set at t-dt
        obs, rew, done, info = self._propagate_realtime_system(ctrl=ctrl)

        # clip next action to action space bounds
        self._cur_action = deepcopy(next_action)
        self._cur_action[K_TURNRATE_CTRL] = np.clip(
            next_action[K_TURNRATE_CTRL], 
            self.action_space[K_TURNRATE_CTRL].low,
            self.action_space[K_TURNRATE_CTRL].high)
        self._cur_action[K_ACCEL_CTRL] = np.clip(
            next_action[K_ACCEL_CTRL], 
            self.action_space[K_ACCEL_CTRL].low,
            self.action_space[K_ACCEL_CTRL].high)

        # assert self.action_space.contains(next_action)

        # return 
        return obs, rew, done, info

    def _propagate_realtime_system(self, ctrl:ArrayLike):
        ''' Advance sim time and propagate dynamics based on elapsed time since last propagation

        Note that a control Lyapunov function (CLF) that automatically steers the system to the goal is 
        part of the environment. If no explicit control is given, then the environments CLF
        takes control

        Note that, by design, this function should do most of the "heavy lifting" of compute in
        the environment. This is because the real-time nature of this environment is such that
        we don't want to chew up compute on the environment side; instead allowing maximum
        utilization of compute by the agent. This function "pauses" sim time while it 
        does it's compute, thus minimizing the impact on agent metrics

        Args:
            ctrl : ArrayLike or None (len=2)
                control vector [dtheta, dv]
                dtheta = turn rate [rad/sec]
                dv = linear acceleration [m/s/s]
                if ctrl is None, then solve control lyapunov QP for stabilizing control

        Returns:
            obs : gym.spaces.?
                observation of state of system
            rew : float
                reward value for current timestep
            done : boolean
                True if episode is done
            info : dict
                dictionary of auxillary information
        '''

        # record wall-clock elapsed time and accumulate simulation time since last update
        wall_lap_time = time.time() - self._wall_clock_sync_time
        sim_lap_time = wall_lap_time*self._time_accel_factor

        # formulate lap time vector for physics propagation
        tvec = np.linspace(0, sim_lap_time, PROPAGATE_TIMESTEPS, endpoint=True)

        # compute CLF control if no external control given
        if ctrl is None:
            ctrl_n_del = self._solve_default_ctrl_clf_qp(
                state=self.__state,
                target=[self._goal.xc, self._goal.yc],
                vmin=self.state_space.low[SS_VIND], vmax=self.state_space.high[SS_VIND],
                u1min=self.action_space[K_TURNRATE_CTRL].low[0], u1max=self.action_space[K_TURNRATE_CTRL].high[0],
                u2min=self.action_space[K_ACCEL_CTRL].low[0], u2max=self.action_space[K_ACCEL_CTRL].high[0],
                gamma_vmin=self.gamma_vmin, gamma_vmax=self.gamma_vmax,
                lambda_Vtheta=self.lambda_Vtheta, lambda_Vspeed=self.lambda_Vspeed,
                p_Vtheta=self.p_Vtheta, p_Vspeed=self.p_Vspeed
            )
            ctrl = np.array(ctrl_n_del[:2]).reshape(2,)
        else:
            # record length of sim time spent in active control
            self._active_ctrl_sim_time += sim_lap_time

        # randomized control disturbances
        dist = np.zeros(4,)
        dist[0] = normal(self.__dist.ctrl.x_mean, self.__dist.ctrl.x_std)
        dist[1] = normal(self.__dist.ctrl.y_mean, self.__dist.ctrl.y_std)
        dist[2] = normal(self.__dist.ctrl.theta_mean, self.__dist.ctrl.theta_std)
        dist[3] = normal(self.__dist.ctrl.v_mean, self.__dist.ctrl.v_std)

        # perform physics propagation
        state_traj = odeint(self.__ode_dubins4d_truth, self.__state, tvec, args=(ctrl,dist))
        if np.any(np.isnan(state_traj)):
            raise ValueError("Physics propagation failed resulting in NaN states\n"+
                "init state: {}\nctrl: {}\nstate traj{}".format(self.__state, ctrl, state_traj))

        # check goal and obstacle collisions
        gcol_any, _, gcol_edge = self._goal.check_traj_intersection(state_traj)
        ocol_any, _, ocol_edge = self._obstacle.check_traj_intersection(state_traj)

        # reward is sparse [-1,0,1] if goal is reached before obstacle
        rew = 0
        if gcol_any:
            if not ocol_any:
                rew = 1
            else:
                # inspect which was encountered first: goal or obstacle
                first_goal_edge = np.where(gcol_edge)[0][0]
                first_obst_edge = np.where(ocol_edge)[0][0]
                if first_goal_edge < first_obst_edge:
                    rew = 1
                else:
                    rew = -1
        elif not gcol_any and ocol_any:
            rew = -1
        else:
            rew = 0
        self._cum_reward += rew

        # update system state
        self.__state = state_traj[-1,:]

        # updtate sim clock time 
        self._wall_clock_elapsed_time += wall_lap_time
        self._sim_time += sim_lap_time

        # check for episode termination conditions (goal or obstacle intersection)
        timeout = self._sim_time >= self._max_episode_sim_time
        done = gcol_any or ocol_any or timeout

        # get observation
        obs = self._get_observation()

        # get auxillary information
        info = self._get_info(done)

        # add a frame to the render collection
        self.renderer.render_step()

        # reset wall clock sync time for next loop
        self._wall_clock_sync_time = time.time()

        return obs, rew, done, info

    def _get_observation(self):
        '''formats observation of system according to observation space

        Returns:
            [0] sim-time [s]
            [1] x-rel-to-goal [m]
            [2] y-rel-to-goal [m]
            [3] heading [rad]
            [4] speed [m/s]
            [5:5+N_RAYS] ray-casts to obstacles [m]
        '''
        obs = [
            self._sim_time,
            self._goal.xc - self.__state[SS_XIND],
            self._goal.yc - self.__state[SS_YIND],
            self.__state[SS_THETAIND],
            self.__state[SS_VIND],
            ]

        obs_ray = self._n_rays*[None]
        for i in range(self._n_rays):
            # get evenly spaced angles relative to vehicle heading
            rel_angle = 2*np.pi/self._n_rays * i
            abs_angle = rel_angle + self.__state[SS_THETAIND]

            # create start point of ray at vehicle location
            abs_start_pt = Point(self.__state[SS_XIND], self.__state[SS_YIND])

            # get endpoints of ray if no collision in abs coords
            abs_end_x = self._max_ray_length*np.cos(abs_angle) + self.__state[SS_XIND]
            abs_end_y = self._max_ray_length*np.sin(abs_angle) + self.__state[SS_YIND]
            abs_end_pt = Point(abs_end_x, abs_end_y)

            # create Linestring from start to end to represent ray
            ray = LineString([abs_start_pt, abs_end_pt])

            # intersect LineString with obstacle Polygon
            if ray.intersects(self._obstacle.polygon):
                ray_obst_inter = ray.intersection(self._obstacle.polygon)

                # get distance from point to intersection
                obs_ray[i] = abs_start_pt.distance(ray_obst_inter)

            else:
                obs_ray[i] = self._max_ray_length

        observation = np.concatenate((obs, obs_ray), dtype=self.observation_space.dtype)
        # if not self.observation_space.contains(observation):
        #     raise ValueError(
        #         "Observation outside of observation space\n"+
        #         "observation >= observation_space.low: {}\n".format(np.greater_equal(observation, self.observation_space.low)) +
        #         "observation <= observation_space.high: {}".format(np.less_equal(observation, self.observation_space.high))
        #     )

        return observation


    def _get_info(self, done:bool):
        '''packagage auxillary info into dictionary, particularly when episode is done'''
        info = dict()
        if done:
            info[K_N_ENV_STEPS] = self._n_env_steps
            info[K_N_ACTIVE_CTRL_ENV_STEPS] = self._n_active_ctrl_env_steps
            info[K_CUM_ACTIVE_CTRL_SIM_TIME] = self._active_ctrl_sim_time
            info[K_CUM_WALL_CLOCK_TIME] = self._wall_clock_elapsed_time
            info[K_CUM_SIM_TIME] = self._sim_time
            info[K_AVG_POLICY_COMPUTE_TIME] = self._wall_clock_elapsed_time/self._n_env_steps
            info[K_CUM_REWARD] = self._cum_reward

        return info


    @staticmethod
    def _solve_default_ctrl_clf_qp(
        state:ArrayLike, 
        target:ArrayLike, 
        vmin, vmax,
        u1min, u1max,
        u2min, u2max,
        gamma_vmin, gamma_vmax,
        lambda_Vtheta, lambda_Vspeed,
        p_Vtheta, p_Vspeed) -> ArrayLike:
        '''Defines and solves quadratic program for dubins4d control lyapunov functions

        This solver acts as the default/passive/inactive controller of the system
        when an active external control algorithm does NOT steer the system

        Think of this as the human driver of the car for which the safety controller needs
        to override in order to avoid obstacles

        Note: by design this controller has no regard for the obstacles in the environment;
        it is the responsibility of the environment solution algorithms to avoid such 
        obstacles
        
        Args:
            state : ArrayLike (len=4)
                state of dubins4d system [x, y, theta, v]
            target : ArrayLike (len=2)
                desired [x,y] to steer system toward
            vmin, vmax  : float
                min and max constraint on speed state variable
            u1min, u1max : float
                min and max constraint on turning rate control variable
            u2min, u2max : float
                min and max constraint on linear acceleration control variable
            gamma_vmax, gamma_vmin : float
                parameter lower-bounding evolution of speed barrier function
            lambda_Vtheta, lambda_Vspeed : float
                parameter upper-bounding evolution of headding and speed lyapunov functions
            p_Vtheta, p_Vspeed : float
                penalty in objective function on heading and speed slack variables that relax stability constraints

            
        Returns:
            ctrl_del : ArrayLike (len=4)
                control variables for dubins4d and slack variables [u1, u2, del_Vtheta, del_Vspeed]
                where del_Vtheta and del_Vspeed are slack variables to relax heading and 
                speed stability constraints (lyapunov functions), respectively

        Notes:
            Assumes dubins-like control-affine system with 
            state: s=[x, y, theta, v], 
            controls: u =[dtheta, dv] 
            dynamics: ds = f(x) + g(x)u = [v*cos(thata); v*sin(theta); 0; 0] + [0,0; 0,0; 1,0; 0,1]u

        Refs:
            Xiao, Wei, and Calin Belta. "Control barrier functions for systems with high relative degree." 2019 IEEE 58th conference on decision and control (CDC). IEEE, 2019.
            Section V.B
        '''

        assert len(state) == 4
        assert len(target) == 2
        assert vmin >= 0
        assert vmax >= vmin
        assert u1max >= u1min
        assert u2max >= u2min
        assert gamma_vmax > 0
        assert gamma_vmin > 0
        assert lambda_Vtheta > 0
        assert lambda_Vspeed > 0
        assert p_Vtheta > 0
        assert p_Vspeed > 0

        # unpack state vars for simple handling
        x, y, theta, v = state
        xd, yd = target

        # distance to target
        xhat = xd - x
        yhat = yd - y

        # heuristic that desired speed increase with distance to target
        dist_targ = np.sqrt(xhat*xhat + yhat*yhat)
        vd = min((dist_targ/2)**2, dist_targ/2)

        # map theta into range [-pi,pi] to avoid windup of Vtheta
        theta = (deepcopy(theta) + np.pi) % (2 * np.pi) - np.pi

        # init QP inequality matrices and vector
        G_all = np.empty((0,4))
        h_all = np.empty((0,1))

        ### STATE BOUNDS CONSTRAINTS (control barrier func) ###

        b_vmax = vmax - v
        Lgbvmaxu1 = 0
        Lgbvmaxu2 = -1
        G_vmax = np.reshape([-Lgbvmaxu1, -Lgbvmaxu2, 0, 0], (1,4))
        h_vmax = np.reshape(gamma_vmax * b_vmax, (1,1))

        # compile with other constraints
        G_all = np.concatenate((G_all, G_vmax), axis=0)
        h_all = np.concatenate((h_all, h_vmax), axis=0)

        b_vmin = v - vmin
        Lgbvminu1 = 0
        Lgbvminu2 = 1
        G_vmin = np.reshape([-Lgbvminu1, -Lgbvminu2, 0, 0], (1,4))
        h_vmin = np.reshape(gamma_vmin * b_vmin, (1,1))

        # compile with other constraints
        G_all = np.concatenate((G_all, G_vmin), axis=0)
        h_all = np.concatenate((h_all, h_vmin), axis=0)

        ### CONTROL BOUNDS CONSTRAINTS ###
        G_ctrl = np.array([
            [   1,  0,  0,  0],
            [   -1, 0,  0,  0],
            [   0,  1,  0,  0],
            [   0,  -1, 0,  0]
        ])
        h_ctrl = np.reshape([u1max, -u1min, u2max, -u2min], (4,1))

        # compile with other constraints
        G_all = np.concatenate((G_all, G_ctrl), axis=0)
        h_all = np.concatenate((h_all, h_ctrl), axis=0)

        ### Vtheta: Control Lyapunov Function for heading stabilization (relaxed) constraint ###

        thetad = np.arctan2(yhat,xhat)

        # heuristic to improve control for theta_d near pi
        if theta < 0 and theta > -np.pi and thetad >= np.pi + theta and thetad <= np.pi:
            thetad = -1.5*np.pi
        if theta > 0 and theta < np.pi and thetad <= -np.pi + theta and thetad >= -np.pi:
            thetad = 1.5*np.pi

        Vtheta = (theta - thetad)**2

        # Lie derivative of Vtheta along f(x)
        LfVtheta = 2*v*(theta - thetad)*(xhat*np.sin(theta)-yhat*np.cos(theta))/(xhat*xhat + yhat*yhat)
        # LfVtheta = 0.0

        # Lie derivative of Vtheta along g(x)
        LgVthetau1 = 2*(theta - thetad)
        LgVthetau2 = 0

        # formatted for inequality constraints
        G_Vtheta = np.reshape([LgVthetau1, LgVthetau2, -1, 0], (1,4))   # 4-vector because slack vars. -1 on first slack var
        h_Vtheta = np.reshape(-LfVtheta - lambda_Vtheta*Vtheta, (1,1))

        # compile with other constraints
        G_all = np.concatenate((G_all, G_Vtheta), axis=0)
        h_all = np.concatenate((h_all, h_Vtheta), axis=0)

        ### Vspeed: Control Lyapunov Function for speed stabilization (relaxed) constraint ###

        Vspeed = (v -vd)*82

        # Lie derivative of Vspeed along f(x)
        LfVspeed = 0

        # Lie derivative of Vspeed along g(x)
        LgVspeedu1 = 0
        LgVspeedu2 = 2*(v-vd)

        # formatted for inequality constraints
        G_Vspeed = np.reshape([LgVspeedu1, LgVspeedu2, 0, -1], (1,4)) # 4-vector because slack vars. -1 on second slack var
        h_Vspeed = np.reshape(-LfVspeed - lambda_Vspeed*Vspeed, (1,1))

        # compile with other constraints
        G_all = np.concatenate((G_all, G_Vspeed), axis=0)
        h_all = np.concatenate((h_all, h_Vspeed), axis=0)
        
        ### OBJECTIVE FUNCTION ###

        P_objective = 2*np.array([
            [   1,  0,  0,  0],
            [   0,  1,  0,  0],
            [   0,  0,  p_Vtheta,   0],
            [   0,  0,  0,  p_Vspeed]
        ],dtype='float')

        q_objective = np.zeros((4,1))

        ### SOLVE QUADRATIC PROGRAM ###

        # solve for control variable and slack variables
        ctrl_n_del = cvx_qp_solver(P=P_objective, q=q_objective, G=G_all, h=h_all)

        return ctrl_n_del

    # @staticmethod
    def __ode_dubins4d_truth(self, X:ArrayLike, t:ArrayLike, u:ArrayLike, d:ArrayLike) -> ArrayLike:
        '''dubins vehicle ordinary differential equations

        Truth model for physics propagation. This is in contrast to whatever
        model solution algorithms (e.g. CBFs, HJ-reach, lrmm) use for the system
        
        Args:
            X : array-like (len=4)
                state variable vector [x, y, theta, v]
                x = x-position [m]
                y = y-position [m]
                theta = heading [rad]
                v = linear speed [m/s]
            t : array-like
                time variable
            u : array-like (len=2)
                control vector [dtheta, dv]
                dtheta = turn rate [rad/sec]
                dv = linear acceleration [m/s/s]
            d : ArrayLike (len=4)
                control disturbances / process noise [dx, dy, dtheta, dv]


        Returns:
            dydt : array-like
                array of the time derivative of state vector
        '''

        dXdt = 4*[None]
        dXdt[0] = X[3] * np.cos(X[2]) + d[0]
        dXdt[1] = X[3] * np.sin(X[2]) + d[1]
        dXdt[2] = u[0] + d[2]
        dXdt[3] = u[1] + d[3]
        # physical constraint: speed is non-negative
        if dXdt[3] < 0 and X[3] < self.state_space.low[SS_VIND] + 1e-3:
            dXdt[3] = 0.0
        if dXdt[3] > 0 and X[3] > self.state_space.high[SS_VIND] - 1e-3:
            dXdt[3] = 0.0
        return dXdt

    def render(self):
        # Just return the list of render frames collected by the Renderer.
        return self.renderer.get_renders()

    def map_state_to_render_window(self, x, y):
        xmin = self.state_space.low[SS_XIND]
        xmax = self.state_space.high[SS_XIND]
        ymin = self.state_space.low[SS_YIND]
        ymax = self.state_space.high[SS_YIND]
        rend_x = (x - xmin)/(xmax - xmin) * self.render_window_size
        rend_y = (1 - (y - ymin)/(ymax - ymin)) * self.render_window_size
        return rend_x, rend_y

    def _render_frame(self, mode: str):
        # This will be the function called by the Renderer to collect a single frame.
        assert mode is not None  # The renderer will not call this function with no-rendering.
    
        import pygame # avoid global pygame dependency. This method is not called with no-render.
    
        canvas = pygame.Surface((self.render_window_size, self.render_window_size))
        canvas.fill((255, 255, 255))

        # First we draw the goal
        rend_goal_xy = self.map_state_to_render_window(self._goal.xc, self._goal.yc)
        rend_goal_r = abs(np.array(rend_goal_xy) - self.map_state_to_render_window(self._goal.xc+self._goal.r, self._goal.yc))[0]
        pygame.draw.circle(
            canvas,
            color=(0, 255, 0),
            center=rend_goal_xy,
            radius=rend_goal_r
            # radius=self._goal.r,
        )
        # Now draw obstacle
        rend_obst_xy = self.map_state_to_render_window(self._obstacle.xc, self._obstacle.yc)
        rend_obst_r = abs(np.array(rend_obst_xy) - self.map_state_to_render_window(self._obstacle.xc+self._obstacle.r, self._obstacle.yc))[0]
        pygame.draw.circle(
            canvas,
            color=(255, 0, 0),
            center=rend_obst_xy,
            radius=rend_obst_r,
        )
        # Now we draw the agent
        rend_agnt_xy = self.map_state_to_render_window(self.__state[SS_XIND], self.__state[SS_YIND])
        # rend_agnt_r = abs(np.array(rend_agnt_xy) - self.map_state_to_render_window(self._goal.xc+self._goal.r, self._goal.yc))[0]
        pygame.draw.circle(
            canvas,
            color=(0, 0, 255),
            center=rend_agnt_xy,
            radius=5,
        )
        # Now draw agent heading
        pygame.draw.line(
            canvas,
            color=(0, 0, 255),
            start_pos=rend_agnt_xy,
            end_pos=self.map_state_to_render_window(
                self.__state[SS_XIND] + self.__state[SS_VIND]*np.cos(self.__state[SS_THETAIND]), 
                self.__state[SS_YIND] + self.__state[SS_VIND]*np.sin(self.__state[SS_THETAIND])
            )
        )

        if mode == "human":
            assert self.render_window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.render_window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.render_clock.tick(self.metadata["render_fps"])
        else:  # rgb_array or single_rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        '''Cleanup open resources (e.g. renderer, threads, etc)'''
        if self.window is not None:
            import pygame 
            
            pygame.display.quit()
            pygame.quit()

class CircleRegion:
    '''Object describes circular region in 2D space of specified radius'''
    def __init__(self, xc:float, yc:float, r:float) -> None:
        '''
        Args:
            xc : float
                x-position of center of circle [m]
            yc : float
                y-position of center of circle [m]
            r : float
                radius of circle [m]
        '''
        # assert r > 0
        self._xc = xc
        self._yc = yc
        self.r = r
        self._polygon = None
        self._update_polygon()

    def _update_polygon(self):
        self._polygon = Point(self._xc, self._yc).buffer(self._r)

    @property
    def polygon(self):
        # NOTe: no setter for polygon; 
        # it should always be inferred from updates to other properties
        return self._polygon

    @property
    def xc(self):
        return self._xc

    @xc.setter
    def xc(self, val):
        self._xc = val
        self._update_polygon()

    @property
    def yc(self):
        return self._yc

    @yc.setter
    def yc(self, val):
        self._yc = val
        self._update_polygon()

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, val):
        if np.less_equal(val, 0):
            raise ValueError("expected radius greater than 0, got {}".format(val))
        self._r = val
        self._update_polygon

    def check_traj_intersection(self, traj: ArrayLike) -> Tuple:
        '''check if propagated states path collide with circular region (e.g. goal or obstacle)
        
        Args:
            traj : ArrayLike (mx4)
                trajectory of m states to be checked for collision with obstacle

        Returns:
            : Tuple(bool, ArrayLike (m,), ArrayLike (m-1,)
                boolean whether any collision exists, True if any collision
                array of booleans, one for each state node, True if node in collision
                array of booleans, one for each state-to-state edge, True if edge in collision
        '''

        n_pts = len(traj)
        any_collision = False
        pt_collision = n_pts*[False]
        edge_collision = (n_pts-1)*[False]
        # pt_collision = np.zeros(n_pts)
        # edge_collision = np.zeros(n_pts-1)

        for i in range(n_pts):

            # check point collisions
            distsqr = np.square(traj[i][SS_XIND]-self._xc) + np.square(traj[i][SS_YIND]-self._yc)
            if np.less_equal(distsqr, np.square(self._r)):
                pt_collision[i] = True
                any_collision = True

            # check edge collisions
            if i == n_pts-1:
                break
            else:
                # create line string for edge
                edge = LineString([
                    (traj[i][SS_XIND], traj[i][SS_YIND]), 
                    (traj[i+1][SS_XIND], traj[i+1][SS_YIND])
                ])
                if self._polygon.intersects(edge):
                    edge_collision[i] = True
                    any_collision = True

        return any_collision, pt_collision, edge_collision

def cvx_qp_solver(P, q, G, h):
    '''Solves quadratic programming using cvxopt

    Args:
        P : ArrayLike (mxm)
            quadratic term in optimization objective
        q : ArrayLike (mx1)
            linear term in optimization objective
        G : ArrayLike (?xm)
            Left-hand side of (<=) inequality constraints
        h : ArrayLike (?x1)
            Right-hand side of (<=) inequality constraints
    
    Ref: 
        https://www.cvxpy.org/examples/basic/quadratic_program.html
        https://cvxopt.org/examples/tutorial/qp.html
    '''
    mat_P = matrix(P)
    mat_q = matrix(q)
    mat_G = matrix(G)
    mat_h = matrix(h)

    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_P, mat_q, mat_G, mat_h)
    return sol['x']
