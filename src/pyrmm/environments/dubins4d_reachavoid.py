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

        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError

    def _get_observation(self):
        raise NotImplementedError

    def _get_info(self):
        '''Get information about the environment that is constant through episode (e.g. control bounds)
        '''
        raise NotImplementedError

    def close(self):
        '''Cleanup open resources (e.g. renderer, threads, etc'''
        raise NotImplementedError

