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

from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv

class HJReachDubins4dReachAvoidAgent():
    def __init__(self,
        grid: Grid,
        dynamics: DubinsCar4D,
        goal: CylinderShape,
        obstacle: CylinderShape,
        time_grid: ArrayLike):

        # specify action space from environment
        self.action_space = deepcopy(Dubins4dReachAvoidEnv.action_space)

        # instantiate properties used to solve for HJI value function
        self._grid = grid
        self._dynamics = dynamics
        self._goal = goal
        self._obstacle = obstacle
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
    def obstacle(self):
        return self._obstacle
    @obstacle.setter
    def obstacle(self, new_obstacle:CylinderShape):
        self._obstacle = new_obstacle
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
        self._hji_values = HJSolver(
            dynamics_obj=self._dynamics, 
            grid=self._grid,
            multiple_value=[self._goal, self._obstacle],
            tau=self._time_grid,
            compMethod={ "TargetSetMode": "minVWithVTarget","ObstacleSetMode": "maxVWithObstacle"},
            plot_option=PlotOptions(do_plot=False, plotDims=[0,1,3]),
            saveAllTimeSteps=True)

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

        # Compute value function at furthest time-horizon at current state
        V_X0 = interpn(self._grid, self._hji_values[..., 0], state)

        # Determine if active control is to be applied 

        # If state is in backward reachable set (with finite time horizon)
        # of obstacle space (i.e. value function <= 0), then compute
        # and employ optimal evasive control
        pass