"""
Functions for defining and using a 4-dimesnional Dubins-like vehicle 
    with state [x-pos, y-pos, heading, speed] and 
    control [turnrate, acceleration]

Ref: 
    Xiao, Wei, and Calin Belta. "High order control barrier functions." IEEE Transactions on Automatic Control (2021).
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9516971&casa_token=rPbyx4QzfZ4AAAAA:EMnVQyqds-d3By8ZP3D1CHwZu3rDzPLMHhvQlDGi8yDSE9ns0M9VgtCD7YjgiTtlhpakh_Yy&tag=1
    Section V.B

Ref:
    https://github.com/SFU-MARS/optimized_dp/blob/4724edaa408832fe2b125fc68ded9d60fdf3a458/odp/dynamics/DubinsCar4D2.py#L19
    https://github.com/SFU-MARS/optimized_dp/blob/master/examples/examples.py

Ref:
    A similar, 3D version of the dynamics are given in:
    Bajcsy, Andrea, et al. "An efficient reachability-based framework for provably safe autonomous navigation in unknown environments." 
    2019 IEEE 58th Conference on Decision and Control (CDC). IEEE, 2019.
"""

import numpy as np
from ompl import base as ob

class Dubins4DStateSpace(ob.CompoundStateSpace):
    """OMPL representation of Dubins 4D state with x-pos, y-pos, heading, speed"""
    def __init__(self, bounds=None):
        """
        Args:
            bounds : Dict
                dictionary of state space bounds. 
                If None, no bounds set (but this causes sampler to sample trivial states)
                If not None, then all entries must be given
                xpos_low : float
                    minimum position states [m]
                xpos_high : float
                    maximum position states [m]
                ypos_low : float
                    minimum position states [m]
                ypos_high : float
                    maximum position states [m]
                speed_low : float
                    minimum velocity states [m/s]
                speed_high : float
                    maximum velocity states [m/s]
        """
        super().__init__()
        self.addSubspace(ob.RealVectorStateSpace(2), 1.0)    # xy-position [m]
        self.addSubspace(ob.SO2StateSpace(), 1.0)            # heading (theta)   [rad]
        self.addSubspace(ob.RealVectorStateSpace(1), 1.0)    # linear speed (v)  [m/s]

        if bounds is not None:

            # ensure bounds properly configured
            req_keys = ['xpos_low', 'xpos_high', 'ypos_low', 'ypos_high', 'speed_low', 'speed_high']
            assert all([k in bounds.keys() for k in req_keys])

            # set state space bounds inherited from environment
            pos_bounds = ob.RealVectorBounds(2)
            pos_bounds.setLow(0, bounds['xpos_low'])
            pos_bounds.setHigh(0, bounds['xpos_high'])
            pos_bounds.setLow(1, bounds['ypos_low'])
            pos_bounds.setHigh(1, bounds['ypos_high'])
            self.getSubspace(0).setBounds(pos_bounds)

            speed_bounds = ob.RealVectorBounds(1)
            speed_bounds.setLow(0, bounds['speed_low'])
            speed_bounds.setHigh(0, bounds['speed_high'])
            self.getSubspace(2).setBounds(speed_bounds)

def ode_dubins4d(y, t, u, vbounds):
    '''dubins vehicle ordinary differential equations
    
    Args:
        y : array-like (len=4)
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
        vbounds : array-like (len=2)
            linear speed bounds [v_min, v_max] [m/s]

    Returns:
        dydt : array-like
            array of the time derivative of state vector
    '''

    dydt = 4*[None]
    dydt[0] = y[3] * np.cos(y[2])
    dydt[1] = y[3] * np.sin(y[2])
    dydt[2] = u[0]
    dydt[3] = u[1]

    # physical constraint: speed is non-negative
    if dydt[3] < 0 and y[3] < vbounds[0] + 1e-3:
        dydt[3] = 0.0
    if dydt[3] > 0 and y[3] > vbounds[1]- 1e-3:
        dydt[3] = 0.0
    return dydt

    return dydt
