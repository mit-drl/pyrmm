import pytest
import pathlib

from pyrmm.setups.single_integrator import SingleIntegrator1DSetup

def test_SingleIntegrator1DPPMSetup_init_0():
    '''Check setup can be initialized without error'''
    SingleIntegrator1DSetup(0, 1, 0, 1)