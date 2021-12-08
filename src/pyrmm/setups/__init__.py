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
    def __init__(self, state_space, control_space, state_validity_fn, propagator_fn):
        ''' create SimpleSetup object
        Args:
            state_space : ob.StateSpace
                state space in which risk metrics are to be evaluated
            control_space : oc.ControlSpace
                control space of system
            state_validity_fn : callable
                decides whether a given state from a specific StateSpace is valid
            propagator_fn : callable
                returns state obtained by applying a control to some arbitrary initial state

        '''
        
        # define the simple setup class
        self.ssetup = oc.SimpleSetup(control_space) 

        # set state validity checker
        validityChecker = ob.StateValidityCheckerFn(partial(
            state_validity_fn, 
            self.ssetup.getSpaceInformation()
        ))
        self.ssetup.setStateValidityChecker(validityChecker)

        # set state-control propagator
        self.ssetup.setStatePropagator(oc.StatePropagatorFn(propagator_fn))
        
