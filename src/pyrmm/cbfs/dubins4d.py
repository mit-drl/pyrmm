"""
Source code for defining control barrier functions (CBF) and control lyapunov functions (CLF)
    for the dubins4d system.
    Also defines quadratic program and solver for CBF+CLF problem for dubins4d system
"""

import numpy as np

from numpy.typing import ArrayLike
from typing import List

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

def cbf_clf_qp(
    state:ArrayLike, 
    target:ArrayLike, 
    obstacles:List[CircleRegion],
    vmin, vmax,
    u1min, u1max,
    u2min, u2max,
    alpha_p1, alpha_p2, alpha_q1, alpha_q2,
    gamma_vmin, gamma_vmax) -> ArrayLike:
    '''Defines and solves quadratic program for dubins4d CBF+CLF with circular obstacles
    
    Args:
        state : ArrayLike
            state of dubins4d system [x, y, theta, v]
        target : ArrayLike
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

        
    Returns:
        ctrl : ArrayLike
            control variables for dubins4d from solution to QP

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
    assert vmin > 0
    assert vmax >= vmin
    assert u1max >= u1min
    assert u2max >= u2min
    assert alpha_p1 > 0
    assert alpha_p2 > 0
    assert alpha_q1 >= 1
    assert alpha_q2 >= 1
    assert gamma_vmax > 0
    assert gamma_vmin > 0

    # unpack state vars for simple handling
    x, y, theta, v = state
    xd, yd, vd = target

    
    ### OBSTACLE SAFETY CONSTRAINTS (control barrier func) ###

    # init QP inequality matrices and vector for safety constraints
    G_safety = np.empty((0,4))
    h_safety = np.empty((0,1))
    
    for obs in obstacles:

        # control barrier function
        b = (x - obs.xc)**2 + (y - obs.yc)**2 - obs.r**2

        # 1st order Lie derivative along f(x)
        Lfb = 2*v*((x-obs.xc)*np.cos(theta) + (y-obs.yc)*np.sin(theta))

        # 2nd order Lie derivative along f(x)
        Lf2b = 2*v**2

        # Cross Lie derivatives along g(x) and f(x) product with u
        LgLfbu1 = 2*v*((y-obs.yc)*np.cos(theta) - (x-obs.xc)*np.sin(theta))
        LgLfbu2 = 2*((x-obs.xc)*np.cos(theta) + (y-obs.yc)*np.sin(theta))

        # Higher order terms
        Lfa1p0 = alpha_q1*alpha_p1*Lfb * b**(alpha_q1-1)
        a2p1 = alpha_p2 * (Lfb + alpha_p1 * b**alpha_q1)**alpha_q2

        # form QP inequality matrices and vectors
        cur_G_safety = np.reshape([-LgLfbu1, -LgLfbu2, 0, 0], (1,4))   # 4 vector because slack variables
        cur_h_safety = np.reshape(Lf2b + Lfa1p0 + a2p1, (1,1))
        G_safety = np.concatenate((G_safety,cur_G_safety), axis=0)
        h_safety = np.concatenate((h_safety,cur_h_safety), axis=0)

    ### STATE BOUNDS CONSTRAINTS (control barrier func) ###
    b_vmax = vmax - v
    G_vmax = np.reshape([0, -1, 0, 0], (1,4))
    h_vmax = np.reshape(gamma_vmax * b_vmax, (1,1))
    b_vmin = v - vmin
    G_vmin = np.reshape([0, 1, 0, 0], (1,4))
    h_vmin = np.reshape(gamma_vmin * b_vmin, (1,1))

    ### CONTROL BOUNDS CONSTRAINTS ###
    pass

    ### STABILIZATION (relaxed) CONSTRAINTS (control lyapunov func) ###
    pass

    ### COMPILE CONSTRAINTS ###
