import pathlib

from pyrmm.setups.dubins import DubinsPPMSetup

def test_DubinsPPMSetup_init_0():
    '''test that DubinsPPMSetup constructed without error'''

    curdir = pathlib.Path(__file__).parent.absolute()
    DubinsPPMSetup(str(curdir.joinpath("border_640x400.ppm")), 1, 1)