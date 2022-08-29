'''
An environment for testing safety-based navigation algorithms on Dubins-like system dynamics.

Using OpenAI Gym API---not because this necessarily a reinforcement learning problem---
but because it offers a convienent, standardized interface for all algorithms
'''

import gym
import time
import numpy as np

from scipy.integrate import odeint
from numpy.random import normal
from numpy.typing import ArrayLike
from types import SimpleNamespace

# state bounds
SB_XMIN = -10.0     # [m]
SB_XMAX = 10.0      # [m]
SB_YMIN = -10.0     # [m]
SB_YMAX = 10.0      # [m]
SB_THETAMIN = -np.inf   # [rad]
SB_THETAMAX = np.inf    # [rad]
SB_VMIN = 0.0       # [m/s]
SB_VMAX = 2.0       # [m/s]

# control bounds
CB_DTHETAMIN = -0.2 # [rad/s]    
CB_DTHETAMAX = 0.2  # [rad/s]
CB_DVMIN = -0.5     # [m/s/s]
CB_DVMAX = 0.5      # [m/s/s]

# number of time steps to analyze per system propagation
PROPAGATE_TIMESTEPS = 16

class Dubins4dReachAvoidEnv(gym.Env):

    def __init__(self):

        # define state space bounds
        self.state_space = gym.spaces.Box(
            low = np.array((SB_XMIN, SB_YMIN, SB_THETAMIN, SB_VMIN)),
            high =  np.array((SB_XMAX, SB_YMAX, SB_THETAMAX, SB_VMAX))
        )

        # define observation space (state space distinct but related to observation space)

        # define action space

        # control and observaiton disturbances 
        # (private because solution algorithms should not know them)
        self.__dist = SimpleNamespace()
        self.__dist.ctrl = SimpleNamespace()
        self.__dist.ctrl.x_mean = 0.0
        self.__dist.ctrl.x_std = 0.1
        self.__dist.ctrl.y_mean = 0.0
        self.__dist.ctrl.y_std = 0.1
        self.__dist.ctrl.theta_mean = 0.0
        self.__dist.ctrl.theta_std = 0.001
        self.__dist.ctrl.v_mean = 0.0
        self.__dist.ctrl.v_std = 0.01

        # setup renderer
        # TODO: see https://www.gymlibrary.dev/content/environment_creation/

        # reset env to generate all instance attributes
        self.reset()

    def reset(self, seed=None):
        # seed random number generator self.np_random
        super().reset(seed=seed)

        # randomize initial state (x [m], y [m], theta [rad], v [m/s])
        # (private because solution algs should not know this explicitly)
        self.__state = np.array([0, 0, 0, 0])

        # randomize goal, obstacle, and vehicle params (speed and control constraints)

        # reset sim clock and sim-to-wall clock sync point
        self.sim_time = 0.0
        # self.sim_lap_time = 0.0
        self.wall_clock_sync_time = time.time()

        # return initial observation and information
        # TODO

    def step(self, action):
        raise NotImplementedError

    def close(self):
        '''Cleanup open resources (e.g. renderer, threads, etc)'''
        raise NotImplementedError

    def _propagate_realtime_system(self, ctrl:ArrayLike):
        ''' Advance sim time and propagate dynamics based on elapsed time since last propagation

        Args:
            ctrl : array-like (len=2)
                control vector [dtheta, dv]
                dtheta = turn rate [rad/sec]
                dv = linear acceleration [m/s/s]
        '''

        # accumulate simulation time since last update
        sim_lap_time = time.time() - self.wall_clock_sync_time

        # formulate lap time vector for physics propagation
        tvec = np.linspace(0, sim_lap_time, PROPAGATE_TIMESTEPS, endpoint=True)

        # perform physics propagation
        state_traj = odeint(self.__ode_dubins4d_truth, self.__state, tvec, args=(ctrl,))

        # check for episode termination conditions (goal or obstacle intersection)
        # TODO

        # update system state
        self.__state = state_traj[-1,:]

        # updtate sim clock time 
        self.sim_time += sim_lap_time

        # reset wall clock sync time for next loop
        self.wall_clock_sync_time = time.time()
        

    def _get_observation(self):
        '''formats observation of system according to observation space, adding observation noise'''
        raise NotImplementedError

    def _get_info(self):
        raise NotImplementedError

    def _check_collisions(self):
        '''check if propagated states path collide with obstacle'''
        raise NotImplementedError

    def __ode_dubins4d_truth(self, X:ArrayLike, t:ArrayLike, u:ArrayLike) -> ArrayLike:
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

        Returns:
            dydt : array-like
                array of the time derivative of state vector
        '''

        dXdt = 4*[None]
        dXdt[0] = X[3] * np.cos(X[2]) + normal(self.__dist.ctrl.x_mean, self.__dist.ctrl.x_std)
        dXdt[1] = X[3] * np.sin(X[2]) + normal(self.__dist.ctrl.y_mean, self.__dist.ctrl.y_std)
        dXdt[2] = u[0] + normal(self.__dist.ctrl.theta_mean, self.__dist.ctrl.theta_std)
        dXdt[3] = u[1] + normal(self.__dist.ctrl.v_mean, self.__dist.ctrl.v_std)
        return dXdt