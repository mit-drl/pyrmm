import pytest
import numpy as np
from functools import partial
from hypothesis import given, settings, strategies as st

from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup

@pytest.fixture
def dummy_r2_setup():
    return get_dummy_r2_setup()

def get_dummy_r2_setup():
    '''define a system in R2 state space with trivial functions'''

    # class prop_cls(oc.StatePropagator):
    #     def __init__(self, spaceInformation):
    #         super().__init__(spaceInformation)
    #     def propagate_path(self, **kwargs):
    #         pass

    # def control_ompl_to_numpy(self, omplCtrl, npCtrl):
    #     return np.zeros(2)
    def control_numpy_to_ompl(npCtrl, omplCtrl):
        """convert 2d control from numpy array to ompl control object in-place
        """
        omplCtrl[0] = npCtrl[0]
        omplCtrl[1] = npCtrl[1]

    def state_numpy_to_ompl(npState, omplState):
        """convert 2d state from numpy array to ompl control object in-place
        """
        omplState[0] = npState[0]
        omplState[1] = npState[1]

    # state_space = ob.SO2StateSpace()
    state_space = ob.RealVectorStateSpace(dim=2)
    control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
    si = oc.SpaceInformation(state_space, control_space)
    state_validity_fn = lambda spaceInformation, state: True
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))
    # si.setStatePropagator(prop_cls(si))
    sys_setup = SystemSetup(space_information=si, eom_ode=lambda y, t, u: [y[0], y[1]])
    sys_setup.control_ompl_to_numpy = lambda omplCtrl, npCtrl=None: np.array([omplCtrl[0], omplCtrl[1]])
    # sys_setup.sample_control_numpy = lambda: np.random.uniform(size=(2,))
    sys_setup.control_numpy_to_ompl = control_numpy_to_ompl
    sys_setup.state_ompl_to_numpy = lambda omplState, npState=None: np.array([omplState[0], omplState[1]])
    sys_setup.state_numpy_to_ompl = state_numpy_to_ompl
    return sys_setup


def test_SystemSetup_init_0(dummy_r2_setup):
    ''' just check that you can build a SystemSetup object
    '''
    pass


def test_SystemSetup_init_raises_0():
    ''' check that init SystemSetup without validity checker and eom raises error
    '''

    state_space = ob.SO2StateSpace()
    control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
    si = oc.SpaceInformation(state_space, control_space)

    # calling SystemSetup constructor should raise error because no 
    # state validity checker or state propagator
    with pytest.raises(AttributeError):
        SystemSetup(si, eom_ode=lambda y, t, u: [y,t,u])

    # now add state validity checker
    state_validity_fn=lambda spaceInformation, state: True
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))

    # this should still raise error because eom not callable
    with pytest.raises(AttributeError):
        SystemSetup(si, eom_ode=True)

    # this should still raise error because eom not have correct number of inputs
    with pytest.raises(AttributeError):
        SystemSetup(si, eom_ode=lambda y, t: [y,t])

    # now add EOMs, should no longer raise error
    SystemSetup(si, eom_ode=lambda y, t, u: [y,t,u])

def test_SystemSeutp_isPathValid_0(dummy_r2_setup):
    '''check that an always valid state give valid path'''

    ds = dummy_r2_setup
    si = ds.space_info

    for _ in range(100):
        # ~~~ ARRANGE ~~~
        # create random path of random length
        nsteps = np.random.randint(low=1, high=100)
        pth = oc.PathControl(si)
        for i in range(nsteps):
            pth.append(si.allocState())

        # ~~~ ACT ~~~
        # check path is valid
        is_valid = ds.isPathValid(pth)

        # ~~~ ASSERT ~~~
        assert is_valid

def test_SystemSeutp_isPathValid_1(dummy_r2_setup):
    '''check that invalid states gives invalid path'''

    # ~~~ ARRANGE ~~~
    ds = dummy_r2_setup
    si = ds.space_info

    # set state validity checker to always return false
    state_validity_fn=lambda spaceInformation, state: False
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))

    for _ in range(100):
        # create random path of random length
        nsteps = np.random.randint(low=1, high=100)
        pth = oc.PathControl(si)
        for i in range(nsteps):
            pth.append(si.allocState())

        # ~~~ ACT ~~~
        # check path is valid
        is_valid = ds.isPathValid(pth)

        # ~~~ ASSERT ~~~
        assert not is_valid


def test_SystemSeutp_isPathValid_2(dummy_r2_setup):
    '''check that one invalid states gives invalid path'''

    # ~~~ ARRANGE ~~~
    ds = dummy_r2_setup
    si = ds.space_info
    assert si.getStateDimension() == 2
    invalid_state_x = 10.599098945310937

    # set state validity checker to always return false
    state_validity_fn=lambda spaceInformation, state: False if np.isclose(state[0],invalid_state_x, atol=1e-9, rtol=1e-9) else True
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))

    for _ in range(100):
        # create random path of random length
        nsteps = np.random.randint(low=1, high=100)
        pth = oc.PathControl(si)
        for i in range(nsteps):
            pth.append(si.allocState())

        # randomly decide to set an invalid state or not
        set_invalid = True if np.random.rand() > 0.5 else False
        if set_invalid:
            pth.getStates()[np.random.randint(0, nsteps)][0] = invalid_state_x

        # ~~~ ACT ~~~
        # check path is valid
        is_valid = ds.isPathValid(pth)

        # ~~~ ASSERT ~~~
        if set_invalid:
            assert not is_valid
        else:
            # extremely low probability that a randomly allocated state would have the invalid x
            assert is_valid

@given(
    x0 = st.floats(allow_nan=False, allow_infinity=False),
    y0 = st.floats(allow_nan=False, allow_infinity=False),
    dur = st.floats(min_value=1e-3, allow_nan=False, allow_infinity=False),
    brnch = st.integers(min_value=1, max_value=4),
    dpth = st.integers(min_value=0, max_value=4),
    stps = st.integers(min_value=2, max_value=8)
)
@settings(deadline=None)
def test_hypothesis_SystemSetup_estimateRiskMetric_0(x0, y0, dur, brnch, dpth, stps):
    '''check that always-valid states always evaluate to 0 risk'''
    
    # ~~~ ARRANGE ~~~
    # create R2 system setup in non-fixture fashion to avoid hypothesis's non-reset of fixtures
    ds = get_dummy_r2_setup()
    si = ds.space_info
    s0 = si.allocState()
    s0[0] = x0; s0[1] = y0
    
    # ~~~ ACT ~~~
    r_s0, _, _ = ds.estimateRiskMetric(
        state = s0, 
        trajectory = None, 
        distance = dur,
        branch_fact = brnch,
        depth = dpth,
        n_steps = stps
    )

    # ~~~ ASSERT ~~~
    assert np.isclose(r_s0, 0.0)

@given(
    x0 = st.floats(allow_nan=False, allow_infinity=False),
    y0 = st.floats(allow_nan=False, allow_infinity=False),
    dur = st.floats(min_value=1e-3, allow_nan=False, allow_infinity=False),
    brnch = st.integers(min_value=1, max_value=4),
    dpth = st.integers(min_value=0, max_value=4),
    stps = st.integers(min_value=2, max_value=8)
)
def test_hypothesis_SystemSetup_estimateRiskMetric_1(x0, y0, dur, brnch, dpth, stps):
    '''check that always-invalid states always evaluate to 1 risk'''
    
    # ~~~ ARRANGE ~~~
    # create R2 system setup in non-fixture fashion to avoid hypothesis's non-reset of fixtures
    ds = get_dummy_r2_setup()
    si = ds.space_info
    
    # set state validity checker to always return false
    state_validity_fn=lambda spaceInformation, state: False
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))

    # generate the initial state
    s0 = si.allocState()
    s0[0] = x0; s0[1] = y0
    
    # ~~~ ACT ~~~
    r_s0, _, _ = ds.estimateRiskMetric(
        state = s0, 
        trajectory = None, 
        distance = dur,
        branch_fact = brnch,
        depth = dpth,
        n_steps = stps
    )

    # ~~~ ASSERT ~~~
    assert np.isclose(r_s0, 1.0)

def test_SystemSetup_estimateRiskMetric_deterministic_0():
    '''check risk metric for a known number of invalid states'''

    # ~~~ ARRANGE ~~~
    ds = get_dummy_r2_setup()
    si = ds.space_info

    # pre-specify state and samples
    x0 = 0.0; y0 = 0.0
    s0 = si.allocState()
    s1 = si.allocState()
    s2 = si.allocState()
    s3 = si.allocState()
    c = si.allocControl()
    dur = 1.0
    s0[0] = x0; s0[1] = y0
    s1[0] = x0+1; s1[1] = y0
    s2[0] = x0-1; s2[1] = y0
    s3[0] = x0; s3[1] = y0-1
    p1 = oc.PathControl(si)
    p2 = oc.PathControl(si)
    p3 = oc.PathControl(si)
    p1.append(s0, c, dur); p1.append(s1)
    p2.append(s0, c, dur); p2.append(s2)
    p3.append(s0, c, dur); p3.append(s3)
    samples = [p1, p2, p3]

    ## set state validity checker to always return false
    state_validity_fn=lambda spaceInformation, state: True if np.isclose(state[1], y0) else False
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))

    # ~~~ ACT ~~~
    r_s0, _ , _ = ds.estimateRiskMetric(
        state = s0, 
        trajectory = None, 
        distance = 1.0,
        branch_fact = 3,
        depth = 1,
        n_steps = 2,
        samples=samples
    )

    # ~~~ ASSERT ~~~
    assert np.isclose(r_s0, 1.0/3.0)

def test_SystemSetup_estimateRiskMetric_deterministic_1():
    '''check minimum risk control is properly returned'''

    # ~~~ ARRANGE ~~~
    ds = get_dummy_r2_setup()
    si = ds.space_info

    # pre-specify state samples
    x0 = 0.0; y0 = 0.0
    s0 = si.allocState()
    s1 = si.allocState()
    s2 = si.allocState()
    s3 = si.allocState()
    s0[0] = x0; s0[1] = y0
    s1[0] = x0+1; s1[1] = y0-1
    s2[0] = x0-1; s2[1] = y0+1
    s3[0] = x0; s3[1] = y0

    # specify controls
    c1 = si.allocControl()
    c2 = si.allocControl()
    c3 = si.allocControl()
    c1[0], c1[1] = [0.46885954, 0.24200232]
    c2[0], c2[1] = [0.25406809, 0.54474039]
    c3[0], c3[1] = [0.75440753, 0.11925401]

    # specify control durations
    d1, d2, d3 = [0.72044402, 0.51690959, 0.4763451 ]

    # formulate as path controls
    p1 = oc.PathControl(si)
    p2 = oc.PathControl(si)
    p3 = oc.PathControl(si)
    p1.append(s0, c1, d1); p1.append(s1)
    p2.append(s0, c2, d2); p2.append(s2)
    p3.append(s0, c3, d3); p3.append(s3)
    samples = [p1, p2, p3]

    ## set state validity checker to always return false
    state_validity_fn=lambda spaceInformation, state: True if np.isclose(state[1], y0) else False
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))

    # ~~~ ACT ~~~
    r_s0, min_risk_ctrl , min_risk_ctrl_dur = ds.estimateRiskMetric(
        state = s0, 
        trajectory = None, 
        distance = 1.0,
        branch_fact = 3,
        depth = 1,
        n_steps = 2,
        samples=samples
    )

    # ~~~ ASSERT ~~~
    assert np.isclose(r_s0, 2.0/3.0)
    assert np.allclose(min_risk_ctrl, [c3[0], c3[1]])
    assert np.isclose(min_risk_ctrl_dur, d3)

if __name__ == "__main__":
    test_SystemSetup_estimateRiskMetric_deterministic_0()