import pytest
import pathlib
import numpy as np
from hypothesis import given, settings, strategies as st

from pyrmm.setups.single_integrator import SingleIntegrator1DSetup

def test_SingleIntegrator1DPPMSetup_init_0():
    '''Check setup can be initialized without error'''
    SingleIntegrator1DSetup(0, 1, 0, 1)

def test_SingleIntegrator1DPPMSetup_estimateRiskMetric_0():
    '''Check risk metric for known, fixed speed, deterministic sytem'''
    
    # ~~~ ARRANGE ~~~
    ds = SingleIntegrator1DSetup(1, 1, 0, 2)
    s0 = ds.space_info.allocState()
    s0[0] = 1.0

    # distance and depth does not reach boundary
    assert np.isclose(0.0,
        ds.estimateRiskMetric(
            state=s0,
            trajectory=None,
            distance=0.2,
            branch_fact=8,
            depth=4,
            n_steps=2
        )
    )

    # distance and depth barely not reach boundary
    assert np.isclose(0.0,
        ds.estimateRiskMetric(
            state=s0,
            trajectory=None,
            distance=0.25 - 1e-6,
            branch_fact=8,
            depth=4,
            n_steps=2
        )
    )

    # distance and depth barely passes boundary
    assert np.isclose(1.0,
        ds.estimateRiskMetric(
            state=s0,
            trajectory=None,
            distance=0.25 + 1e-6,
            branch_fact=8,
            depth=4,
            n_steps=2
        )
    )

@settings(deadline=500)
@given(
    x0 = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    v = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    lb = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    ub_del = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    dur = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    brch = st.integers(min_value=1, max_value=8),
    dpth = st.integers(min_value=0, max_value=8),
    nstp = st.integers(min_value=2, max_value=8)
)
def test_hypothesis_SingleIntegrator1DPPMSetup_estimateRiskMetric_0(x0, v, lb, ub_del, dur, brch, dpth, nstp):
    '''Check risk metric for known, fixed speed, deterministic sytem'''

    # ~~~ ARRANGE ~~~
    ub = lb + ub_del
    ds = SingleIntegrator1DSetup(min_speed=v, max_speed=v, lower_bound=lb, upper_bound=ub)
    s0 = ds.space_info.allocState()
    s0[0] = x0

    # ~~~ ACT ~~~
    r = ds.estimateRiskMetric(
        state=s0,
        trajectory=None,
        distance=dur,
        branch_fact=brch,
        depth=dpth,
        n_steps=nstp
    )    

    # ~~~ ASSERT ~~~
    xf = x0 + v * dur * dpth
    if x0 < lb or x0 > ub or xf < lb or xf > ub:
        assert np.isclose(r, 1.0)
    else:
        assert np.isclose(r, 0.0)


