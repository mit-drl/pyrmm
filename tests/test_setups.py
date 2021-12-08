from pyrmm.setups import SystemSetup
from functools import partial
from ompl import base as ob
from ompl import control as oc

def test_SystemSetup_init_0():
    ''' just check that you can build a SystemSetup object
    '''

    class prop_cls(oc.StatePropagator):
        def __init__(self, spaceInformation):
            super().__init__(spaceInformation)


    state_space = ob.SO2StateSpace()
    control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
    si = oc.SpaceInformation(state_space, control_space)
    state_validity_fn=lambda spaceInformation, state: True
    si.setStateValidityChecker(ob.StateValidityCheckerFn(partial(state_validity_fn, si)))
    si.setStatePropagator(prop_cls(si))
    SystemSetup(space_information=si)
