import pathlib

from pyrmm.setups.dubins import DubinsPPMSetup

from ompl import base as ob

PPM_FILE_0 = str(pathlib.Path(__file__).parent.absolute().joinpath("border_640x400.ppm"))

def test_DubinsPPMSetup_init_0():
    '''test that DubinsPPMSetup constructed without error'''
    DubinsPPMSetup(PPM_FILE_0, 1, 1)

def test_DubinsPPMSetup_state_checker_0():
    '''test that known states validity'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_FILE_0, 1, 1)
    valid_fn = ds.ssetup.getStateValidityChecker()
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(300)
    s0().setY(200)
    s0().setYaw(0)
    s1 = ob.State(ob.DubinsStateSpace())
    s1().setX(700)
    s1().setY(200)
    s1().setYaw(0)

    # ~~~ ACT ~~~
    s0_valid = valid_fn.isValid(s0())
    s1_valid = valid_fn.isValid(s1())

    # ~~~ ASSERT ~~~
    assert s0_valid
    assert not s1_valid
