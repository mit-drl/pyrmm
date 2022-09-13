# Class for defining HJ-reachability-based agent to navigate the dubins4d_reachavoid enviornment

import numpy as np

from copy import deepcopy
from typing import List
from numpy.typing import ArrayLike
from scipy.interpolate import interpn

from odp.Grid import Grid
from odp.dynamics import DubinsCar4D
from odp.Shapes import CylinderShape
from odp.Plots import PlotOptions
from odp.solver import HJSolver, computeSpatDerivArray

from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv, K_ACTIVE_CTRL, K_TURNRATE_CTRL, K_ACCEL_CTRL

class HJReachDubins4dReachAvoidAgent():
    def __init__(self,
        grid: Grid,
        dynamics: DubinsCar4D,
        goal: CylinderShape,
        obstacles: List[CylinderShape],
        time_grid: ArrayLike):

        # specify action space from environment
        self.action_space = deepcopy(Dubins4dReachAvoidEnv.action_space)

        # instantiate properties used to solve for HJI value function
        self._grid = grid
        self._dynamics = dynamics
        self._goal = goal
        self._obstacles = obstacles
        self._time_grid = time_grid

        # Solve for HJI value function on discrete state space grid
        self.update_hji_values()

    @property
    def hji_values(self):
        return self._hji_values
    @hji_values.setter
    def hji_values(self,val):
        # a property with no setter since it should always
        # be calculate from other properties
        raise NotImplementedError('HJI Values should only be implicitly set by call to update_hji_values()')

    @property
    def grid(self):
        return self._grid
    @grid.setter
    def grid(self, new_grid:Grid):
        self._grid = new_grid
        self.update_hji_values()

    @property
    def dynamcis(self):
        return self._dynamics
    @dynamcis.setter
    def dynamics(self, new_dynamics:DubinsCar4D):
        self._dynamics = new_dynamics
        self.update_hji_values()

    @property
    def goal(self):
        return self._goal
    @goal.setter
    def goal(self, new_goal:CylinderShape):
        self._goal = new_goal
        self.update_hji_values()

    @property
    def obstacles(self):
        return self._obstacles
    @obstacles.setter
    def obstacles(self, new_obstacles:List[CylinderShape]):
        self._obstacles = new_obstacles
        self.update_hji_values()

    @property
    def time_grid(self):
        return self._time_grid
    @time_grid.setter
    def time_grid(self, new_time_grid):
        self._time_grid = new_time_grid
        self.update_hji_values()

    def update_hji_values(self):
        '''solve HJI value function on discrete grid'''
        # self._hji_values = HJSolver(
        #     dynamics_obj=self._dynamics, 
        #     grid=self._grid,
        #     multiple_value=[self._goal, self._obstacle],
        #     tau=self._time_grid,
        #     compMethod={ "TargetSetMode": "minVWithVTarget","ObstacleSetMode": "maxVWithObstacle"},
        #     plot_option=PlotOptions(do_plot=False, plotDims=[0,1,3]),
        #     saveAllTimeSteps=True)
        self._hji_values = HJSolver(
            dynamics_obj=self._dynamics, 
            grid=self._grid,
            multiple_value=self._obstacles,
            tau=self._time_grid,
            compMethod={ "TargetSetMode": "minVWithVTarget"},
            plot_option=PlotOptions(do_plot=False, plotDims=[0,1,3]),
            saveAllTimeSteps=True)

    def opt_ctrl(self, spat_deriv):
        ''' compute optimal control as function of spatial derivatve
        
        Args:
            grad_Vf : ArrayLike 
                spatial derivative (gradient) at particular state in [x,y,v,theta] order
        
        Returns
            opt_a : float
                optimal acceleration control
            opt_w : float
                optimal turnrate control
        '''
        opt_a = self.dynamics.uMax[0]
        opt_w = self.dynamics.uMax[1]

        # The initialized control only change sign in the following cases
        if self.dynamics.uMode == "min":
            if spat_deriv[2] > 0:
                opt_a = self.dynamics.uMin[0]
            if spat_deriv[3] > 0:
                opt_w = self.dynamics.uMin[1]
        else:
            if spat_deriv[2] < 0:
                opt_a = self.dynamics.uMin[0]
            if spat_deriv[3] < 0:
                opt_w = self.dynamics.uMin[1]

        return opt_a, opt_w

    def get_action(self, state: ArrayLike):
        '''given environment state, determine appropriate action

        Args:
            state : ArrayLike
                state of system in [x, y, v, theta] ordering
                x = x-position [m]
                y = y-position [m]
                v = linear speed [m/s]
                theta = heading [rad] 
        
        Ref:
            Bajcsy, Andrea, et al. "An efficient reachability-based framework for 
                provably safe autonomous navigation in unknown environments." 
                2019 IEEE 58th Conference on Decision and Control (CDC). IEEE, 2019.
        '''

        # create an action object for modification from sample
        action = self.action_space.sample()

        # extract hji value function on state space grid at furthest time horizon
        Vf = self._hji_values[..., 0]

        # check if state is within solution grid
        if not np.all([s>=g[0] and s<=g[-1] for s,g in zip(state, self._grid.grid_points)]):
            # state outside of HJ PDE solution space, do not take active control
            action[K_ACTIVE_CTRL] = False
            return action

        # compute value function at furthest time-horizon at current state
        Vf_state = interpn(self._grid.grid_points, Vf, state)

        # determine if active control is to be applied
        if Vf_state > 0.0:
            # state is in safe set, do not use active control
            action[K_ACTIVE_CTRL] = False
            return action

        # if state is in backward reachable set (with finite time horizon)
        # of obstacle space (i.e. value function <= 0), then compute
        # and employ optimal evasive control
        action[K_ACTIVE_CTRL] = True

        # Compute spatial derivatives of final value function at every discrete state on grid
        delVf_delx = computeSpatDerivArray(self._grid, Vf, deriv_dim=1, accuracy="low")
        delVf_dely = computeSpatDerivArray(self._grid, Vf, deriv_dim=2, accuracy="low")
        delVf_delv = computeSpatDerivArray(self._grid, Vf, deriv_dim=3, accuracy="low")
        delVf_delth = computeSpatDerivArray(self._grid, Vf, deriv_dim=4, accuracy="low")

        # interpolate spatial derivative at state
        grad_Vf_state = (
            interpn(self._grid.grid_points, delVf_delx, state), 
            interpn(self._grid.grid_points, delVf_dely, state),
            interpn(self._grid.grid_points, delVf_delv, state),
            interpn(self._grid.grid_points, delVf_delth, state)
        )

        # compute optimal control
        opt_a, opt_w = self.opt_ctrl(grad_Vf_state)
        action[K_ACCEL_CTRL] = opt_a
        action[K_TURNRATE_CTRL] = opt_w

        return action

