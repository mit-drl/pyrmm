from pyrmm.setups import SystemSetup

from ompl import base as ob
from ompl import control as oc

def test_SystemSetup_init_0():
    ''' just check that you can build a SystemSetup object
    '''
    state_space = ob.SO2StateSpace()
    control_space = oc.RealVectorControlSpace(stateSpace=state_space, dim=2)
    SystemSetup(
        state_space=state_space, 
        control_space=control_space, 
        state_validity_fn=lambda spaceInformation, state: True, 
        propagator_fn=lambda start, control, duration, state: None)
