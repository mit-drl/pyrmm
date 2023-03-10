'''
collection of modules used to create OMPL SimpleSetup objects 
for different systems that can be used to compute risk metric
maps
'''
import inspect
import numpy as np
from scipy.integrate import odeint

from numpy.typing import ArrayLike

from ompl import control as oc

class SystemSetup:
    ''' generic class for building SimpleSetup objects for risk metric map analysis

    Refs:
        https://ompl.kavrakilab.org/api_overview.html
    '''

    # Define observation shape as a class property 
    # that must be implemented by child class
    observation_shape = NotImplemented
    
    def __init__(
            self, 
            space_information, 
            eom_ode: callable, 
            risk_fn: callable = np.mean):
        ''' create SimpleSetup object
        Args:
            space_information : ob.SpaceInformation OR oc.SpaceInformation
                state and control space info for which risk metrics are to be evaluated
            eom_ode : callable
                ordinary diff eq defining system's equations of motion
                function with strictly 3 input arguments
                y : ArrayLike
                    state variable vector
                t : ArrayLike
                    time variable
                u : ArrayLike
                    control vector
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
        
        # ensure that a ompl state propagator has NOT been set
        if self.space_info.getStatePropagator() is not None:
            raise AttributeError("OMPL state propagator class deprecated in favor of propagate_path function!")
        
        # ensure that equations of motion ODE have been defined
        self.eom_ode = eom_ode
        
        # equations of motion can be set to None if child class
        # implements propagate_path. If not None, check 
        # that eoms are callable and have appropriate inputs
        if self.eom_ode is not None:
            if not callable(self.eom_ode):
                raise AttributeError("Equations of motion ODE must be callable")
            
            # ensure eoms only have three input arguments: [y, t, u]
            # which correspond to state, time, and control vectors.
            # This ensurese the EOM it is compatible with scipy.odeint API 
            # Other parameters of EOMS should be defined by using a lambda function
            # to make a parameterized function have only the three above variables
            if len(inspect.getfullargspec(self.eom_ode)[0]) != 3:
                raise AttributeError("callable equation of motion ODE must only have 3 input arguments!")

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
            samples : list(PathControl)
                list of sampled states with trajectories to those states
        '''

        # re-seed numpy to generate different seeds in parallel processes to avoid control sample repeats
        np.random.seed()

        if policy == 'uniform_random':

            # access space information
            si = self.space_info

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
                c = si.allocControl()
                np_c = self.sample_control_numpy()
                self.control_numpy_to_ompl(npCtrl=np_c, omplCtrl=c)

                # propagate sampled control
                # si.getStatePropagator().propagate_path(
                self.propagate_path(
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
    
    def state_ompl_to_numpy(self, omplState, npState=None):
        """convert ompl state to numpy array

        Args:
            omplState : ob.State
                ompl state object
            npState : ArrayLike OR None
                state represented in numpy array
                if None, return np array, otherwise modify in place
        """
        raise NotImplementedError('To be implemented by child class')
    
    def state_numpy_to_ompl(self, npState, omplState):
        """convert state from numpy array to ompl object in-place

        Args:
            npState : ArrayLike
                state represented in numpy array
            omplState : ob.State
                ompl state object to be modified in place
        """
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

    def control_numpy_to_ompl(self, npCtrl, omplCtrl):
        """convert control from numpy array to ompl control object in-place


        Args:
            npCtrl : ArrayLike
                control represented as np array
            omplCtrl : oc.Control
                OMPL control object to be modified in-place
                https://ompl.kavrakilab.org/classompl_1_1control_1_1ControlSpace.html
        """
        raise NotImplementedError('To be implemented by child class')
    
    @staticmethod
    def path_numpy_to_ompl(
        np_states: ArrayLike, 
        np_controls: ArrayLike, 
        np_times: ArrayLike, 
        omplPath,
        state_numpy_to_ompl_func: callable,
        control_numpy_to_ompl_func: callable,
        ):
        """ Convert numpy-like description of a trajectory into ompl PathControl object (in place)

        Args:
            np_states : ArrayLike (n,nStateDims)
                array of n states in path in numpy-like format
            np_controls : ArrayLike (n-1,nCtrlDims)
                array of n-1 control inputs along path in numpy-like format
            np_times : ArrayLike (n,)
                array of n time steps of path in numpy-like format
            omplPath : oc.PathControl
                ompl PathControl object, modified in-place
            state_numpy_to_ompl_func: callable
                function for converting state from numpy array to ompl object
            control_numpy_to_ompl_func: callable
                function for converting control from numpy array to ompl object
        
        Returns:
            None
        """
        nsteps = len(np_states)
        assert nsteps == omplPath.getStateCount() == len(np_times)

        # store each intermediate point in the solution as pat of the path
        pstates = omplPath.getStates()
        pcontrols = omplPath.getControls()
        pdurs = omplPath.getControlDurations()
        assert len(pcontrols) == len(pdurs) == len(np_controls) == nsteps-1
        for i in range(nsteps-1):

            # write the state
            state_numpy_to_ompl_func(npState=np_states[i], omplState=pstates[i])

            # write the control 
            control_numpy_to_ompl_func(npCtrl=np_controls[i], omplCtrl=pcontrols[i])

            # write the timing values
            pdurs[i] = np_times[i+1] - np_times[i]
        
        # store final state
        state_numpy_to_ompl_func(npState=np_states[-1], omplState=pstates[-1])

    @staticmethod
    def path_ompl_to_numpy(
        omplPath, 
        np_states: ArrayLike, 
        np_controls: ArrayLike, 
        np_times: ArrayLike,
        state_ompl_to_numpy_func: callable,
        control_ompl_to_numpy_func: callable,
        ):
        """ Convert ompl PathControl object to numpy arrays in-place

        Args:
            omplPath : oc.PathControl
                ompl PathControl object, modified in-place
            np_states : ArrayLike (n,nStateDims)
                array of n states in path in numpy-like format
            np_controls : ArrayLike (n-1,nCtrlDims)
                array of n-1 control inputs along path in numpy-like format
            np_times : ArrayLike (n,)
                array of n time steps of path in numpy-like format
            state_ompl_to_numpy_func: callable
                function for converting state from ompl object to numpy array
            control_ompl_to_numpy_func: callable
                function for converting control from ompl object to numpy array
        
        Returns:
            None (conversion performed in-place)
        """

        nsteps = len(np_states)
        assert nsteps == omplPath.getStateCount() == len(np_times)

        # store each intermediate point in the solution as pat of the path
        pstates = omplPath.getStates()
        pcontrols = omplPath.getControls()
        pdurs = omplPath.getControlDurations()
        assert len(pcontrols) == len(pdurs) == len(np_controls) == nsteps-1
        np_times[0] = 0.0
        for i in range(nsteps-1):

            # write the state
            state_ompl_to_numpy_func(omplState=pstates[i], npState=np_states[i])

            # write the control 
            control_ompl_to_numpy_func(omplCtrl=pcontrols[i], npCtrl=np_controls[i])

            # write the timing values
            np_times[i+1] = pdurs[i] + np_times[i]
        
        # store final state
        state_ompl_to_numpy_func(omplState=pstates[-1], npState=np_states[-1])

    def sample_control_numpy(self):
        """Randomly sample valid control in numpy format using numpy random

        Note: an error was found where the OMPL-provided control sampler,
        SpaceInformation.allocControlSampler, produced repeated values when 
        run in parallel processes using
        multiprocessing.Pool. No direct fix could be determined so
        this is a workaround that forces control sampling in numpy

        No equivalent bug has yet been identified in state sampling

        Args: 
            None

        Returns:
            np_ctrl : ArrayLike
                the ranomly sampled control in numpy-format (instead of ompl-format)
        """

        cspace = self.space_info.getControlSpace()
        cbounds = cspace.getBounds()
        cdim = cspace.getDimension()
        return np.random.uniform(
            [cbounds.low[i] for i in range(cdim)], 
            [cbounds.high[i] for i in range(cdim)], 
            (cdim,)
        )

    def propagate_path(self, state, control, duration, path):
        """ propagate from start based on control, store path in-place

        Args:
            state : ob.State internal
                start state of propagation
            control : oc.Control
                control to apply during propagation,
                will be bounded by control space bounds
            duration : float
                duration of propagation
            path : oc.ControlPath
                path from state to result in nsteps. initial state is state, final state is result

        Returns:
            None

        Notes:
            This function is similar, but disctinct from 'StatePropagator.propogate', thus its different name to no overload `propagate`. 
            propogate does not store or return the path to get to result
            
            Currently using scipy's odeint. This creates a dependency on scipy and is likely inefficient
            because it's perform the numerical integration in python instead of C++. 
            Could be improved later
        """

        # ensure that equations of motion ODE have been defined
        if self.eom_ode is None:
            raise AttributeError("If equations of motion are None, then child class must implement propagate_path")

        # unpack objects from space information for ease of use
        nstatedims = self.space_info.getStateDimension()
        cspace = self.space_info.getControlSpace()
        nctrldims = cspace.getDimension()
        cbounds = cspace.getBounds()
        nsteps = path.getStateCount()
        assert nsteps >= 2
        assert duration >= 0

        # package init state and time vector
        np_state = self.state_ompl_to_numpy(omplState=state)
        assert len(np_state) == nstatedims, "Inconsistency found between state dims and ompl-to-numpy state conversion"
        
        # create equi-spaced time vector based on number or elements
        # in path object
        t = np.linspace(0.0, duration, nsteps)

        # clip the control to ensure it is within the control bounds
        np_control = self.control_ompl_to_numpy(omplCtrl=control)
        assert len(np_control) == nctrldims, "Inconsistency found between control dims and ompl-to-numpy control conversion"
        np_bounded_control = [np.clip(np_control[i], cbounds.low[i], cbounds.high[i]) for i in range(nctrldims)]

        # call scipy's ode integrator
        sol = odeint(self.eom_ode, np_state, t, args=(np_bounded_control,))

        # expand bounded_control into array for conversion to ompl
        # PathControl object
        ctrl_array = np.array([np_bounded_control for i in range(nsteps-1)])

        self.path_numpy_to_ompl(
            np_states=sol, 
            np_times=t, 
            np_controls=ctrl_array, 
            omplPath=path,
            state_numpy_to_ompl_func=self.state_numpy_to_ompl,
            control_numpy_to_ompl_func=self.control_numpy_to_ompl)