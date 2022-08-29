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
from numpy.random import normal
from numpy.typing import ArrayLike
from types import SimpleNamespace

# state space constants (e.g. vector indexes, bounds)
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

# control bounds
ACTIVE_CTRL = "active_ctrl"
TURNRATE_CTRL = "turnrate_ctrl"
ACCEL_CTRL = "accel_ctrl"
DURATION_CTRL = "duration_ctrl"
# CS_ACTCTRLIND = 0   # index in control space for active control boolean
# CS_DTHETAIND = 1    # index in control space for turn rate
# CS_DVIND = 2        # index in control space for linear accel
# CS_DT = 3           # index in control space for control application time
CS_DTHETAMIN = -0.2 # [rad/s]    
CS_DTHETAMAX = 0.2  # [rad/s]
CS_DVMIN = -0.5     # [m/s/s]
CS_DVMAX = 0.5      # [m/s/s]

# number of time steps to analyze per system propagation
PROPAGATE_TIMESTEPS = 16

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
        assert r > 0
        self.xc = xc
        self.yc = yc
        self.r = r

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

class Dubins4dReachAvoidEnv(gym.Env):

    def __init__(self):

        # define state space bounds
        self.state_space = gym.spaces.Box(
            low = np.array((SS_XMIN, SS_YMIN, SS_THETAMIN, SS_VMIN)),
            high =  np.array((SS_XMAX, SS_YMAX, SS_THETAMAX, SS_VMAX))
        )

        # define observation space (state space distinct but related to observation space)
        # TODO

        # define action space
        self.action_space = gym.spaces.Dict({
            ACTIVE_CTRL: gym.spaces.Discrete(2),
            TURNRATE_CTRL: gym.spaces.Box(low=CS_DTHETAMIN, high=CS_DTHETAMAX),    # [rad/s]
            ACCEL_CTRL: gym.spaces.Box(low=CS_DVMIN, high=CS_DVMAX),         # [m/s/s]
            DURATION_CTRL: gym.spaces.Box(low=0, high=np.inf)
        })

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
        self.goal = CircleRegion(5.0, 0.0, 1.0)

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

        Note that a control Lyapunov function (CLF) that automatically steers the system to the goal is 
        part of the environment. If no explicit control is given, then the environments CLF
        takes control

        Args:
            ctrl : ArrayLike or None (len=2)
                control vector [dtheta, dv]
                dtheta = turn rate [rad/sec]
                dv = linear acceleration [m/s/s]
                if ctrl is None, then solve control lyapunov QP for stabilizing control
        '''

        # accumulate simulation time since last update
        sim_lap_time = time.time() - self.wall_clock_sync_time

        # formulate lap time vector for physics propagation
        tvec = np.linspace(0, sim_lap_time, PROPAGATE_TIMESTEPS, endpoint=True)

        # compute CLF control if no external control given
        if ctrl is None:
            ctrl_del = self._solve_passive_ctrl_clf_qp(
                state=self.__state,
                target=[self.goal.xc, self.goal.yc, 0.0],
                vmin=self.state_space.low[SS_VIND], vmax=self.state_space.high[SS_VIND],
                u1min=self.action_space[TURNRATE_CTRL].low[0], u1max=self.action_space[TURNRATE_CTRL].high[0],
                u2min=self.action_space[ACCEL_CTRL].low[0], u2max=self.action_space[ACCEL_CTRL].low[0],
                gamma_vmin=1, gamma_vmax=1,
                lambda_Vtheta=1, lambda_Vspeed=1,
                p_Vtheta=1, p_Vspeed=1
            )
            ctrl = np.array(ctrl_del[:2]).reshape(2,)

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

    def _solve_passive_ctrl_clf_qp(self,
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
            target : ArrayLike (len=3)
                desired [x,y,v] to steer system toward (note desired theta inferred)
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
        assert len(target) == 3
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
        xd, yd, vd = target

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

        xhat = xd - x
        yhat = yd - y
        thetad = np.arctan2(yhat,xhat)
        Vtheta = (theta - thetad)**2

        # Lie derivative of Vtheta along f(x)
        LfVtheta = 2*v*(theta - thetad)*(xhat*np.sin(theta)-yhat*np.cos(theta))/(xhat*xhat + yhat*yhat)

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
        ctrl_del = cvx_qp_solver(P=P_objective, q=q_objective, G=G_all, h=h_all)

        return ctrl_del

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