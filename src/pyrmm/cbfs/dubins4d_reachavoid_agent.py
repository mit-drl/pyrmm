"""
Source code for defining control barrier functions (CBF) and control lyapunov functions (CLF)
    for the dubins4d system.
    Also defines quadratic program and solver for CBF+CLF problem for dubins4d system
"""

import numpy as np

from copy import deepcopy
from cvxopt import solvers, matrix
from numpy.typing import ArrayLike
from typing import List

from pyrmm.environments.dubins4d_reachavoid import \
    Dubins4dReachAvoidEnv, CircleRegion, cvx_qp_solver, \
    K_ACTIVE_CTRL, K_TURNRATE_CTRL, K_ACCEL_CTRL

class CBFDubins4dReachAvoidAgent():
    def __init__(self,
        goal: CircleRegion,
        obstacles: List[CircleRegion],
        vmin, vmax,
        u1min, u1max,
        u2min, u2max,
        alpha_p1, alpha_p2, alpha_q1, alpha_q2,
        gamma_vmin, gamma_vmax,
        lambda_Vtheta, lambda_Vspeed,
        p_Vtheta, p_Vspeed) -> None:
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

        # specify action space from environment
        self.action_space = deepcopy(Dubins4dReachAvoidEnv.action_space)

        self.goal = goal
        self.obstacles = obstacles
        self.vmin = vmin
        self.vmax = vmax
        self.u1min = u1min
        self.u1max = u1max
        self.u2min = u2min
        self.u2max = u2max
        self.alpha_p1 = alpha_p1
        self.alpha_p2 = alpha_p2
        self.alpha_q1 = alpha_q1
        self.alpha_q2 = alpha_q2
        self.gamma_vmin = gamma_vmin
        self.gamma_vmax = gamma_vmax
        self.lambda_Vtheta = lambda_Vtheta
        self.lambda_Vspeed = lambda_Vspeed
        self.p_Vtheta = p_Vtheta
        self.p_Vspeed = p_Vspeed

    def get_action(self, state: ArrayLike):
        '''given environment state, determine appropriate action

        Args:
            state : ArrayLike
                state of system in [x, y, theta, v] ordering
                x = x-position [m]
                y = y-position [m]
                theta = heading [rad] 
                v = linear speed [m/s]
        
        '''

        # create an action object for modification from sample
        action = self.action_space.sample()
        action[K_ACTIVE_CTRL] = False

        # solve CBF-CLF quadratic program at state
        ctrl_n_del, G_safety, h_safety = self._solve_cbf_clf_qp(
            state = state,
            target = [self.goal.xc, self.goal.yc],
            obstacles = self.obstacles,
            vmin = self.vmin,
            vmax = self.vmax,
            u1min = self.u1min,
            u1max = self.u1max,
            u2min= self.u2min,
            u2max= self.u2max,
            alpha_p1= self.alpha_p1,
            alpha_p2= self.alpha_p2,
            alpha_q1= self.alpha_q1,
            alpha_q2=  self.alpha_q2,
            gamma_vmin= self.gamma_vmin,
            gamma_vmax= self.gamma_vmax,
            lambda_Vtheta= self.lambda_Vtheta,
            lambda_Vspeed= self.lambda_Vspeed,
            p_Vtheta= self.p_Vtheta,
            p_Vspeed= self.p_Vspeed
        )

        # check if feasible solution found
        if ctrl_n_del is not None:
            # check barrier function  for equality; if so take active a control
            if np.any(np.isclose(np.dot(G_safety, ctrl_n_del), h_safety, rtol=1e-2)):
                # at least one safety constraint active, apply active safety control
                action[K_ACTIVE_CTRL] = True
                action[K_TURNRATE_CTRL][0] = ctrl_n_del[0]
                action[K_ACCEL_CTRL][0] = ctrl_n_del[1]

        return action

    @staticmethod
    def _solve_cbf_clf_qp(
        state:ArrayLike, 
        target:ArrayLike, 
        obstacles:List[CircleRegion],
        vmin, vmax,
        u1min, u1max,
        u2min, u2max,
        alpha_p1, alpha_p2, alpha_q1, alpha_q2,
        gamma_vmin, gamma_vmax,
        lambda_Vtheta, lambda_Vspeed,
        p_Vtheta, p_Vspeed) -> ArrayLike:
        '''Defines and solves quadratic program for dubins4d CBF+CLF with circular obstacles
        
        Args:
            state : ArrayLike (len=4)
                state of dubins4d system in [x, y, theta, v] ordering
                x: x-position [m]
                y: y-position [m]
                theta: heading [rad]
                v: linear speed [m/s]
            target : ArrayLike (len=3)
                desired [x,y,v] to steer system toward (note desired theta inferred)
            obstacles : List[CircleRegion]
                list of circular obstacles
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
        assert alpha_p1 > 0
        assert alpha_p2 > 0
        assert alpha_q1 >= 1
        assert alpha_q2 >= 1
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
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # init QP inequality matrices and vector
        G_all = np.empty((0,4))
        h_all = np.empty((0,1))

        ### OBSTACLE SAFETY CONSTRAINTS (control barrier func) ###

        # init QP inequality matrices and vector for safety constraints
        G_safety = np.empty((0,4))
        h_safety = np.empty((0,1))
        
        for obs in obstacles:

            # control barrier function
            b = (x - obs.xc)**2 + (y - obs.yc)**2 - obs.r**2

            if np.less(b, 0.0):
                # barrier function already violate, return None for control
                return None, None, None

            # 1st order Lie derivative along f(x)
            Lfb = 2*v*((x-obs.xc)*np.cos(theta) + (y-obs.yc)*np.sin(theta))

            # 2nd order Lie derivative along f(x)
            Lf2b = 2*v**2

            # Cross Lie derivatives along g(x) and f(x) product with u
            LgLfbu1 = 2*v*((y-obs.yc)*np.cos(theta) - (x-obs.xc)*np.sin(theta))
            LgLfbu2 = 2*((x-obs.xc)*np.cos(theta) + (y-obs.yc)*np.sin(theta))

            # Higher order terms
            Lfa1p0 = alpha_q1*alpha_p1*Lfb * b**(alpha_q1-1)

            temp_var = Lfb + alpha_p1 * b**alpha_q1
            if np.less(temp_var, 0.0):
                # barrier function already violate, return None for control
                return None, None, None
            a2p1 = alpha_p2 * (temp_var)**alpha_q2

            # form QP inequality matrices and vectors
            cur_G_safety = np.reshape([-LgLfbu1, -LgLfbu2, 0, 0], (1,4))   # 4 vector because slack variables
            cur_h_safety = np.reshape(Lf2b + Lfa1p0 + a2p1, (1,1))
            G_safety = np.concatenate((G_safety,cur_G_safety), axis=0)
            h_safety = np.concatenate((h_safety,cur_h_safety), axis=0)

        # compile with other constraints
        G_all = np.concatenate((G_all, G_safety), axis=0)
        h_all = np.concatenate((h_all, h_safety), axis=0)

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

        return ctrl_del, G_safety, h_safety