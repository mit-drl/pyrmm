'''
Create SystemSetup for Dubins-4D car, i.e. 
state : [x, y, heading, speed]
control : [d_theta, d_speed]
'''

from ompl import base as ob

from pyrmm.setups import SystemSetup

class Dubins4DReachAvoidSetup(SystemSetup):
    def __init__(self, env):
        '''
        Args:
            env : Dubins4dReachAvoidEnv
                Instance of Dubins4dReachAvoidEnv
        '''

        # create state space
        state_space = ob.CompoundStateSpace()
        state_space.addSubspace(ob.RealVectorStateSpace(2), 1.0)    # xy-position [m]
        state_space.addSubspace(ob.SO2StateSpace(), 1.0)            # heading (theta)   [rad]
        state_space.addSubspace(ob.RealVectorStateSpace(1), 1.0)    # linear speed (v)  [m/s]

        # set state space bounds from environment
        pos_bounds = ob.RealVectorBounds(2)
        pos_bounds.setLow(0, float(env.state_space.low[0]))
        pos_bounds.setHigh(0, float(env.state_space.high[0]))
        pos_bounds.setLow(1, float(env.state_space.low[1]))
        pos_bounds.setHigh(1, float(env.state_space.high[1]))
        state_space.getSubspace(0).setBounds(pos_bounds)

        speed_bounds = ob.RealVectorBounds(1)
        speed_bounds.setLow(0, float(env.state_space.low[3]))
        speed_bounds.setHigh(0, float(env.state_space.high[3]))
        state_space.getSubspace(2).setBounds(speed_bounds)

        # create control space

        
