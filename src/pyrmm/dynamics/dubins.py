"""
Functions for defining and using Dubins Car dynamics

Ref: http://planning.cs.uiuc.edu/node821.html
Ref: https://github.com/AtsushiSakai/PythonRobotics/tree/master/PathPlanning/DubinsPath
"""
import numpy as np

def dubinsODE(y, t, u, speed):
    '''dubins vehicle ordinary differential equations
    
    Args:
        q : ???
            state variable vector [x, y, theta]
        t : ???
            time variable
        u : np.array
            control vector [dtheta]
        speed : float
            constant tangential speed
    '''

    dydt = 3*[None]
    dydt[0] = speed * np.cos(y[2])
    dydt[1] = speed * np.sin(y[2])
    dydt[2] = u[0]
    return dydt
