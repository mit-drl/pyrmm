'''
collection of modules used to create OMPL SimpleSetup objects 
for different systems that can be used to compute risk metric
maps
'''
import numpy as np
from copy import deepcopy
from functools import partial

from ompl import base as ob
from ompl import control as oc

class SystemSetup:
    ''' generic class for building SimpleSetup objects for risk metric map analysis

    Refs:
        https://ompl.kavrakilab.org/api_overview.html
    '''
    def __init__(self, space_information, risk_fn=np.mean):
        ''' create SimpleSetup object
        Args:
            space_information : ob.SpaceInformation OR oc.SpaceInformation
                state and control space info for which risk metrics are to be evaluated
            risk_fn : callable
                function for evaluating risk metrics (e.g. expected risk, CVaR)
            # state_validity_fn : callable
            #     decides whether a given state from a specific StateSpace is valid
            # propagator_cls : class
            #     returns state obtained by applying a control to some arbitrary initial state

        '''

        # make space_information and risk metric func member attributes
        self.space_info = space_information
        self.risk_fn = risk_fn

        # ensure that a state validity checker has been set
        if self.space_info.getStateValidityChecker() is None:
            raise AttributeError("State validity checker must be set by child class!")

        # ensure that a state propagator has been set
        if self.space_info.getStatePropagator() is None:
            raise AttributeError("State propagator must be set by child class!")

        # ensure that state propagator has a propagator that returns a path
        if not hasattr(self.space_info.getStatePropagator(), 'propagate_path'):
            raise AttributeError("State propagator must implement propagate_path function that computes path to result!")

    def isPathValid(self, path):
        '''check if any state on path is in collision with obstacles
        
        Args:
            path : oc.Path
        
        Returns:
            true if all path states are valid (not colliding with obstacles)

        Notes:
            This serves a very similar (identical?) role as OMPL's MotionValidator,
            however, there currently exists a bug that prevents access to a motion validator
            using the python bindings.
            See: https://github.com/ompl/ompl/issues/860
        '''
        for i in range(path.getStateCount()):
            if not self.space_info.isValid(path.getState(i)):
                return False
        
        return True

    def sampleReachableSet(self, state, distance, n_samples, policy='uniform_random', n_steps=8):
        '''Draw n samples from state space near a given state using a policy

        Args:
            state : ob.State internal
                state for which nearby samples are to be drawn
            distance : float
                state-space-specific distance to sample within
            n_samples : int
                number of samples to draw
            policy : str
                string description of policy to use
                defaults to undirected control sampler
            n_steps : int
                number of discrete steps in path to each sample

        Returns:
            samples : list(ob.State)
                list of sampled states
        '''

        if policy == 'uniform_random':

            # access space information
            si = self.space_info

            # use default undirected control sampler 
            csampler = si.allocControlSampler()
            c = si.allocControl()
            samples = [None]*n_samples

            for i in range(n_samples):

                # allocate a path object for the sample
                # note that allocated states and controls are
                # dummy placeholders that are overwritten in propagate_path
                p = oc.PathControl(self.space_info)
                for j in range(n_steps-1):
                    p.append(state=si.allocState(), control=si.allocControl(), duration=0)
                
                # allocate final state
                p.append(state=si.allocState())
                assert p.getStateCount() == n_steps
                assert p.getControlCount() == n_steps-1

                # sample control input
                csampler.sample(c)

                # propagate sampled control
                si.getStatePropagator().propagate_path(
                    state = state,
                    control = c,
                    duration = distance,
                    path = p
                )

                # assign sampled path to samples list
                samples[i] = p

        else:
            raise NotImplementedError("No reachable set sampling implemented for policy {}".format(policy))

        return samples

    def estimateRiskMetric(self, state, trajectory, distance, branch_fact, depth, n_steps, 
        policy='uniform_random', 
        samples=None,
        base_estimate_fn=None):
        '''Sampling-based, recursive risk metric estimation at specific state
        
        Args:
            state : ob.State
                state at which to evaluate risk metric
            trajectory : oc.PathControl
                trajectory arriving at state 
            distance : double
                state-space-specific distance to sample within
            branch_fact : int
                number of samples to draw
            depth : int
                number of recursive steps to estimate risk
            n_steps : int
                number of intermediate steps in sample paths
            policy : str
                string description of policy to use
            samples : list[oc.PathControl]
                list of pre-specified to samples for deterministic calc
            base_estimate_fn : Callable
                recursion base function to be called to estimate risk and
                min risk trajectory at leaf nodes. Can be used for bootstrapping

        Returns:
            risk_est : float
                coherent risk metric estimate at state
        '''

        # check if state is in collision
        z = not self.space_info.isValid(state)

        # if state is not in collision, check trajectory for collision (if it exists)
        if (not z) and (trajectory is not None):
            z = not self.isPathValid(trajectory)

        # recursion base: state is failure or leaf node without base estimation function
        if z or (depth==0 and base_estimate_fn is None):
            base_ctrl = self.space_info.allocControl()
            for i in range(self.space_info.getControlSpace().getDimension()):
                base_ctrl[i] = 0.0
            return float(z), self.control_ompl_to_numpy(base_ctrl), 0.0

        # recursion base: leaf node with base estimation function
        if depth == 0 and base_estimate_fn is not None:
            return base_estimate_fn(state)

        # sample reachable states
        if samples is None:
            samples = self.sampleReachableSet(
                state=state, 
                distance=distance, 
                n_samples=branch_fact, 
                policy=policy,
                n_steps=n_steps)
        else:
            assert len(samples) == branch_fact

        # recursively compute risk estimates at sampled states
        risk_vals = branch_fact*[None]
        min_risk = np.inf
        min_risk_ctrl = None
        min_risk_ctrl_dur = None
        for i in range(branch_fact):
            risk_vals[i], _, _ = self.estimateRiskMetric(
                state=samples[i].getState(n_steps-1),
                trajectory=samples[i],
                distance=distance,
                branch_fact=branch_fact,
                depth=depth-1,
                n_steps=n_steps,
                policy=policy,
                base_estimate_fn=base_estimate_fn
            )

            # store minimum risk control
            if np.less(risk_vals[i], min_risk):
                min_risk = risk_vals[i]
                min_risk_ctrl_dur = samples[i].getControlDurations()[0]
                min_risk_ctrl = self.control_ompl_to_numpy(samples[i].getControls()[0])

        # Evaluate the risk metric
        return self.risk_fn(risk_vals), min_risk_ctrl, min_risk_ctrl_dur

    def observeState(self, state):
        '''query observation from a particular state
        Args:
            state : ompl.base.State
                state from which to make observation
        Returns:
            observation : ndarray
                array giving observation values
        '''
        raise NotImplementedError('To be implemented by child class')

    def control_ompl_to_numpy(self, omplCtrl, npCtrl=None):
        """ convert OMPL control object to numpy
        Args:
            omplCtrl : oc.Control
                OMPL control object
                https://ompl.kavrakilab.org/classompl_1_1control_1_1ControlSpace.html
            npCtrl : ndarray
                control represented as np array
                if not None, input argument is modified in place, else returned
        """
        raise NotImplementedError('To be implemented by child class')
        
