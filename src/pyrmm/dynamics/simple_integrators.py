''' Dynamics for integrator systems in real vector spaces
'''
from numpy.typing import ArrayLike

def ode_1d_single_integrator(y, t, u):
    ''' equations of motion for 1-Dimensional single integrator
    
    Args:
        y : array-like
            state variable vector [x]
        t : array-like
            time variable
        u : array-like
            control vector [xdot]

    Returns:
        dydt : array-like
            array of the time derivative of state vector
    '''
    dydt = [u[0]]
    return dydt

def ode_1d_double_integrator(y: ArrayLike, t: ArrayLike, u: ArrayLike):
    """ equations of motion for 1-Dimensional double integrator

    Args:
        y : ArrayLike
            state variable vector [pos, vel]
        t : ArrayLike
            time variable
        u : ArrayLike
            control vector [acc]
    """
    dydt = [y[1], u[0]]
    return dydt