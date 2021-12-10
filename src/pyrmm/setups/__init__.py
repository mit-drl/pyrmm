'''
collection of modules used to create OMPL SimpleSetup objects 
for different systems that can be used to compute risk metric
maps
'''

from functools import partial

from ompl import base as ob
from ompl import control as oc

class SystemSetup:
    ''' generic class for building SimpleSetup objects for risk metric map analysis

    Refs:
        https://ompl.kavrakilab.org/api_overview.html
    '''
    def __init__(self, space_information):
        ''' create SimpleSetup object
        Args:
            space_information : ob.SpaceInformation OR oc.SpaceInformation
                state and control space info for which risk metrics are to be evaluated
            # state_validity_fn : callable
            #     decides whether a given state from a specific StateSpace is valid
            # propagator_cls : class
            #     returns state obtained by applying a control to some arbitrary initial state

        '''

        # make space_information a member attribute
        self.space_info = space_information

        # ensure that a state validity checker has been set
        if self.space_info.getStateValidityChecker() is None:
            raise AttributeError("State validity checker must be set by child class!")

        # ensure that a state propagator has been set
        if self.space_info.getStatePropagator() is None:
            raise AttributeError("State propagator must be set by child class!")
        
        # define the simple setup class
        # self.ssetup = oc.SimpleSetup(space_information) 

        # # set state validity checker
        # validityChecker = ob.StateValidityCheckerFn(partial(
        #     state_validity_fn, 
        #     self.ssetup.getSpaceInformation()
        # ))
        # self.ssetup.setStateValidityChecker(validityChecker)

        # # set state-control propagator
        # statePropagator = propagator_cls(self.ssetup.getSpaceInformation())
        # self.ssetup.setStatePropagator(statePropagator)

    def sampleReachableSet(self, state, distance, n_samples, policy='default'):
        '''Draw n samples from state space near a given state using a policy

        Args:
            state : ob.State
                state for which nearby samples are to be drawn
            distance : float
                state-space-specific distance to sample within
            n_samples : int
                number of samples to draw
            policy : str
                string description of policy to use
                defaults to undirected control sampler

        Returns:
            samples : list(ob.State)
                list of sampled states
        '''

        if policy == 'default':

            # access space information
            si = self.space_info

            # use default undirected control sampler 
            csampler = si.allocControlSampler()
            c = si.allocControl()
            samples = [None]*n_samples

            for i in range(n_samples):

                # allocate memory for sampled state
                samples[i] = si.allocState() 

                # sample control input
                csampler.sample(c)

                # propagate sampled control
                si.getStatePropagator().propagate(
                    state = state(),
                    control = c,
                    duration = distance,
                    result = samples[i]
                )
        else:
            raise NotImplementedError("No reachable set sampling implemented for policy {}".format(policy))

        return samples
        
