'''
Create SystemSetup for Dubins-4D car, i.e. 
state : [x, y, heading, speed]
control : [d_theta, d_speed]
'''


from pyrmm.setups import SystemSetup

class Dubins4DCircleObstSetup(SystemSetup):
    def __init__(self, 
        v_min: float, v_max: float, 
        dtheta_min: float, dtheta_max: float,
        dv_min: float, dv_max: float):
        '''
        Args:
            v_min : float
                min linear speed [m/s]
            v_max : float
                max linear speed [m/s]
            dthteta_min : float
                min turnrate control bound [rad/s]
            dtheta_max : float
                max turnrate control bound [rad/s]
            dv_min : float 
                min linear acceleration control bound [m/s]
            dv_max : float
                max linear acceleration control bound [m/s]
        '''

        assert v_min >= 0
        assert v_max >= v_min
        assert dtheta_max >= dtheta_min
        assert dv_max >= dv_min

        # save init args for re-creation of object
        self.v_min = v_min
        self.v_max = v_max
        self.dtheta_min = dtheta_min
        self.dtheta_max = dtheta_max
        self.dv_min = dv_min
        self.dv_max = dv_max

        
