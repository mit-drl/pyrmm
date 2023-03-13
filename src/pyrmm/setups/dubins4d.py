'''
Create SystemSetup for Dubins-4D car, i.e. 
state : [x, y, heading, speed]
control : [d_theta, d_speed]
'''

import copyreg
import numpy as np

import pyrmm.dynamics.dubins4d as D4D

from functools import partial
from scipy.integrate import odeint

from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
from pyrmm.environments.dubins4d_reachavoid import K_TURNRATE_CTRL, K_ACCEL_CTRL

class Dubins4dReachAvoidSetup(SystemSetup):
    def __init__(self, env):
        '''
        Args:
            env : Dubins4dReachAvoidEnv
                Instance of Dubins4dReachAvoidEnv
        '''

        # store environment for unpickling
        self.env = env

        # create state space
        sbounds = dict()
        sbounds['xpos_low'] = float(self.env.state_space.low[0])
        sbounds['xpos_high'] = float(self.env.state_space.high[0])
        sbounds['ypos_low'] = float(self.env.state_space.low[1])
        sbounds['ypos_high'] = float(self.env.state_space.high[1])
        sbounds['speed_low'] = float(self.env.state_space.low[3])
        sbounds['speed_high'] = float(self.env.state_space.high[3])
        state_space = D4D.Dubins4dStateSpace(bounds=sbounds)

        # create control space and set bounds inherited from environment
        control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
        cbounds = ob.RealVectorBounds(2)
        cbounds.setLow(0, float(self.env.action_space[K_TURNRATE_CTRL].low[0]))
        cbounds.setHigh(0, float(self.env.action_space[K_TURNRATE_CTRL].high[0]))
        cbounds.setLow(1, float(self.env.action_space[K_ACCEL_CTRL].low[0]))
        cbounds.setHigh(1, float(self.env.action_space[K_ACCEL_CTRL].high[0]))
        control_space.setBounds(cbounds)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)

        # create and set state validity checker
        validityChecker = ob.StateValidityCheckerFn(partial(self.isStateValid, space_info))
        space_info.setStateValidityChecker(validityChecker)

        # call parent init to create simple setup
        super().__init__(
            space_information=space_info,
            eom_ode=lambda y, t, u: D4D.ode_dubins4d(
                y, t, u, [
                    space_info.getStateSpace().getSubspace(2).getBounds().low[0],
                    space_info.getStateSpace().getSubspace(2).getBounds().high[0]
                ]
            )
        )

    def __reduce__(self):
        ''' Function to enable re-creation of unpickable object

        Note: See comments about potential risks here
        https://stackoverflow.com/a/50308545/4055705
        '''
        return (Dubins4dReachAvoidSetup, (self.env,))

    def isStateValid(self, spaceInformation, state):
        ''' check if particular state intersect obstacle using env-specific trajectory checker
        Args:
            spaceInformation : ob.SpaceInformationPtr
                state space information as given by SimpleSetup.getSpaceInformation
            state : ob.State
                state to check for validity
        
        Returns:
            True if state in bound and not in collision with obstacles
        '''

        # convert state to numpy
        np_state = state_ompl_to_numpy(state).reshape(1,4)

        # check obstacle collision
        for obst in self.env._obstacles:
            is_collision, _, _ = obst.check_traj_intersection(np_state)
            if is_collision:
                return False

        # is_valid = is_speed_valid and not is_collision
        return True

    def isPathValid(self, path):
        '''check if path intersects obstacles using env-specific trajectory checker
        
        Args:
            path : oc.Path
                OMPL representation of path to be checked
        
        Returns:
            true if all path states and interpolated lines are valid (not colliding with obstacles)

        Notes:
            This overrides SystemSetup function that just looks at discrete states, not interpolated paths
        '''

        # convert path to array of states
        np_traj = np.array([state_ompl_to_numpy(path.getState(i)) for i in range(path.getStateCount())])

        # check collision validity
        for obst in self.env._obstacles:
            any_collision, _, _ = obst.check_traj_intersection(np_traj)
            if any_collision:
                return False
        
        return True

    def observeState(self, state):
        '''query observation from a particular state
        Args:
            state : ob.State
                state from which to make observation
        Returns:
            observation : ArrayLike
                array giving observation values
        '''
        # convert state to numpy
        np_state = state_ompl_to_numpy(omplState=state)
        return self.env._get_observation(state=np_state)

    def base_risk_estimator(self, state):
        """heuristic for estimate risk at leaf node of tree"""

        # get observation of state
        obs = self.observeState(state)

        # compute stopping time
        v = obs[4]
        
        if v >= 0:
            a_min = self.env.action_space[K_ACCEL_CTRL].low[0]
            assert a_min < 0
        else:
            # this is an edge case where dynamics propagated beyond 
            # the v>0 constraint
            a_min = self.env.action_space[K_ACCEL_CTRL].high[0]
            assert a_min > 0

        t_s = -v/a_min
        assert t_s > 0

        # compute stopping distance
        d_s = 0.5*a_min*t_s*t_s + v*t_s

        if d_s >= 0:
            r0 = obs[5]
        else:
            # negative stopping distance, ie. negative intial velocity
            # is an edge case and uncertain behavior. Use 
            # minimum of all ray casts
            r0 = min(obs[5:])

        # stopping distance as fraction of forward ray cast length
        if r0 <= 0:
            risk_est = 1.0
        else:
            risk_est = min(abs(d_s)/r0, 1.0)
        assert risk_est >= 0 and risk_est <= 1.0

        # min risk control and control duration 
        # (although these are more for consistency in return pattern
        # and should not be directly applied)
        min_risk_ctrl_est = np.array([0, a_min])
        min_risk_ctrl_dur_est = t_s

        return risk_est, min_risk_ctrl_est, min_risk_ctrl_dur_est
    
    def state_ompl_to_numpy(self, omplState, npState=None):
        """redirect to static method"""
        return state_ompl_to_numpy(omplState=omplState, npState=npState)
    
    def state_numpy_to_ompl(self, npState, omplState):
        """redirect to static method"""
        return state_numpy_to_ompl(npState=npState, omplState=omplState)
    
    def control_ompl_to_numpy(self, omplCtrl, npCtrl=None):
        """redirect to static method"""
        return control_ompl_to_numpy(omplCtrl=omplCtrl, npCtrl=npCtrl)
    
    def control_numpy_to_ompl(self, npCtrl, omplCtrl):
        """redirect to static method"""
        return control_numpy_to_ompl(omplCtrl=omplCtrl, npCtrl=npCtrl)

def state_ompl_to_numpy(omplState, npState=None):
    """convert Dubins4d ompl state to numpy array

    Args:
        omplState : ob.CompoundState
            dubins 4d state in ompl CompoundState format
        npState : ndarray (4,)
            dubins 4d state represented in np array in [x,y,theta,v] ordering
    """
    ret = False
    if npState is None:
        npState = np.empty(4,)
        ret = True
    npState[0] = omplState[0][0]
    npState[1] = omplState[0][1]
    npState[2] = omplState[1].value
    npState[3] = omplState[2][0]

    if ret:
        return npState

def state_numpy_to_ompl(npState, omplState):
    """convert dubins4d state in numpy array in [x,y,theta,v] to ompl compound state

    Args:
        npState : ndarray (4,)
            dubins 4d state represented in np array in [x,y,theta,v] ordering
        omplState : ob.CompoundState
            dubins 4d state in ompl CompoundState format
    """
    omplState[0][0] = npState[0]
    omplState[0][1] = npState[1]
    omplState[1].value = npState[2]
    omplState[2][0] = npState[3]

def control_ompl_to_numpy(omplCtrl, npCtrl=None):
        """convert Dubins4d ompl control object to numpy array

        Args:
            omplCtrl : oc.Control
                dubins4d control in ompl RealVectorControl format
            npCtrl : ndarray (2,)
                dubins4d control represented in np array with [turnrate, acceleration] ordering
                if not None, input argument is modified in place, else returned
        """
        ret = False
        if npCtrl is None:
            npCtrl = np.empty(2,)
            ret = True

        npCtrl[0] = omplCtrl[0]
        npCtrl[1] = omplCtrl[1]

        if ret:
            return npCtrl
        
def control_numpy_to_ompl(npCtrl, omplCtrl):
        """convert dubins4d control from numpy array to ompl control object in-place

        Args:
            npCtrl : ndarray (2,)
                dubins4d control represented in np array with [turnrate, acceleration] ordering
            omplCtrl : oc.Control
                dubins4d control in ompl RealVectorControl format
        """

        assert npCtrl.shape == (2,), "Unexpected shape {}".format(npCtrl.shape)
        omplCtrl[0] = npCtrl[0]
        omplCtrl[1] = npCtrl[1]

_DUMMY_DUBINS4DSTATESPACE = D4D.Dubins4dStateSpace()

def _pickle_Dubins4dState(state):
    '''pickle Dubins4d state (OMPL compound state) object'''
    px = state[0][0]
    py = state[0][1]
    theta = state[1].value
    v = state[2][0]
    return _unpickle_Dubins4dState, (px,py,theta,v)

def _unpickle_Dubins4dState(px, py, theta, v):
    '''unpickle Dubins4d state (OMPL compound state) object'''
    state = _DUMMY_DUBINS4DSTATESPACE.allocState()
    state[0][0] = px
    state[0][1] = py
    state[1].value = theta
    state[2][0] = v
    return state

def update_pickler_dubins4dstate():
    '''updates pickler to enable pickling and unpickling of ompl objects'''
    copyreg.pickle(_DUMMY_DUBINS4DSTATESPACE.allocState().__class__, _pickle_Dubins4dState, _unpickle_Dubins4dState)