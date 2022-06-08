import numpy as np

from hypothesis import strategies as st
from hypothesis import given

import pyrmm.utils.utils as U

def test_check_collision_circular_obstacles_simple():
    # check for collision with a single obstacle at origin
    a = U.Node2D(0,0)
    o = [(0, 0, 1)]
    assert U.check_collision_2D_circular_obstacles(a,o)

@given(
    st.floats(min_value=0.0, max_value=np.pi, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=2*np.pi, allow_nan=False, allow_infinity=False)
)
def test_hypothesis_spherical_to_cartesian_origin(theta, phi):
    '''check that origin in spherical returns origin in cartesian'''
    # ~~ ARRANGE ~~
    rho = 0.0

    # ~~ ACT ~~
    x, y, z = U.spherical_to_cartesian(rho, theta, phi)

    # ~~ ASSERT ~~
    assert np.isclose(x, 0,0)
    assert np.isclose(y, 0,0)
    assert np.isclose(z, 0,0)

@given(
    st.floats(min_value=0.0, max_value = 1e8, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=2*np.pi, allow_nan=False, allow_infinity=False)
)
def test_hypothesis_spherical_to_cartesian_xyplane(rho, phi):
    '''check that 90 deg theta always lands on xyplane'''
    # ~~ ARRANGE ~~
    theta = np.pi/2

    # ~~ ACT ~~
    x, y, z = U.spherical_to_cartesian(rho, theta, phi)

    # ~~ ASSERT ~~
    assert np.isclose(x, rho*np.cos(phi))
    assert np.isclose(y, rho*np.sin(phi))
    assert np.isclose(0.0, z)

@given(
    st.integers(min_value=1, max_value=1e3),
    st.integers(min_value=0, max_value=1e6)
)
def test_hypothesis_min_linfinity_int_vector(n,b):
    # ~~ ARRANGE ~~
    # ~~ ACT ~~
    x = U.min_linfinity_int_vector(n,b)

    # ~~ ASSERT ~~
    assert len(x) == n
    assert np.sum(x) == b
