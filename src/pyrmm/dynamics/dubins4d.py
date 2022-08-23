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

def ode_dubins4d(y, t, u):
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

    Returns:
        dydt : array-like
            array of the time derivative of state vector
    '''

    dydt = 4*[None]
    dydt[0] = y[3] * np.cos(y[2])
    dydt[1] = y[3] * np.sin(y[2])
    dydt[2] = u[0]
    dydt[3] = u[1]
    return dydt
