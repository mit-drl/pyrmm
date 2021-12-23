import pytest
import numpy as np
from functools import partial

from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup

@pytest.fixture
def dummy_r2_setup():
    '''define a system in R2 state space with trivial functions'''

    class prop_cls(oc.StatePropagator):
        def __init__(self, spaceInformation):
            super().__init__(spaceInformation)
        def propagate_path():
            pass

    # state_space = ob.SO2StateSpace()
    state_space = ob.RealVectorStateSpace(dim=2)
    control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
    si = oc.SpaceInformation(state_space, control_space)
    state_validity_fn=lambda spaceInformation, state: True
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))
    si.setStatePropagator(prop_cls(si))
    return SystemSetup(space_information=si)


def test_SystemSetup_init_0(dummy_r2_setup):
    ''' just check that you can build a SystemSetup object
    '''
    pass


def test_SystemSetup_init_raises_0():
    ''' check that init SystemSetup without validity checker and propagator raises error
    '''

    state_space = ob.SO2StateSpace()
    control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
    si = oc.SpaceInformation(state_space, control_space)

    # calling SystemSetup constructor should raise error because no 
    # state validity checker or state propagator
    with pytest.raises(AttributeError):
        SystemSetup(si)

    # now add state validity checker
    state_validity_fn=lambda spaceInformation, state: True
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))

    # this should still raise error because no propagator
    with pytest.raises(AttributeError):
        SystemSetup(si)

    # now add propagator, should no longer raise error
    class prop_cls(oc.StatePropagator):
        def __init__(self, spaceInformation):
            super().__init__(spaceInformation)
        def propagate_path():
            pass
    si.setStatePropagator(prop_cls(si))
    SystemSetup(si)

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