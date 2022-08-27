'''
An environment for testing safety-based navigation algorithms on Dubins-like system dynamics.

Using OpenAI Gym API---not because this necessarily a reinforcement learning problem---
but because it offers a convienent, standardized interface for all algorithms
'''

import gym

class Dubins4dReachAvoidEnv(gym.Env):
    def __init__(self):

        # define observation space

        # define action space

        # setup renderer
        # TODO: see https://www.gymlibrary.dev/content/environment_creation/

        raise NotImplementedError()

    def reset(self, seed=None):
        # seed random number generator self.np_random
        super().reset(seed=seed)

        # randomize goal, obstacle, and vehicle params (speed and control constraints)

        # reset sim clock

        # return initial observation and information

        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError

    def observe_system(self):
        ''' Advance sim time and propagate dynamics based on elapsed time since last observation
        '''
        raise NotImplementedError

    def _get_observation(self):
        '''formats observation of system according to observation space'''
        raise NotImplementedError

    def _get_info(self):
        raise NotImplementedError

    def check_collisions(self):
        '''check if propagated states path collide with obstacle'''
        raise NotImplementedError

    def close(self):
        '''Cleanup open resources (e.g. renderer, threads, etc)'''
        raise NotImplementedError

