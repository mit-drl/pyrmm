''' Dynamics for integrator systems in real vector spaces
'''

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