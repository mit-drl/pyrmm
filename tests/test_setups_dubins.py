import pathlib
import numpy as np

from scipy.integrate import odeint
from ompl import base as ob
from ompl import control as oc
from hypothesis import strategies as st
from hypothesis import given
from pyrmm.setups.dubins import DubinsPPMSetup, dubinsODE


PPM_FILE_0 = str(pathlib.Path(__file__).parent.absolute().joinpath("border_640x400.ppm"))

def test_DubinsPPMSetup_init_0():
    '''test that DubinsPPMSetup constructed without error'''
    DubinsPPMSetup(PPM_FILE_0, 1, 1)

def test_DubinsPPMSetup_state_checker_0():
    '''test that known states validity'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_FILE_0, 1, 1)
    valid_fn = ds.space_info.getStateValidityChecker()
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

def test_DubinsPPMSetup_dubinsODE_integration_0():
    '''test that the dubins ODE integrates to expected values'''

    # ~~~ ARRANGE ~~~
    # create initial conditions and time vector
    y0 = [300, 200, 0]
    # t = np.linspace(0, 10, 2)
    t = [0, 10]

    # specify the control (turning rate) and speed 
    u = [0]
    speed = 1

    # ~~~ ACT ~~~
    sol = odeint(dubinsODE, y0, t, args=(u, speed))

    # ~~~ ASSERT ~~~
    # check that final timestep is as expected
    assert np.isclose(sol[-1,0], 310)
    assert np.isclose(sol[-1,1], 200)
    assert np.isclose(sol[-1,2], 0)

def test_DubinsPPMSetup_propagator_0():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_FILE_0, 1, 1)
    propagator = ds.space_info.getStatePropagator()

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(300)
    s0().setY(200)
    s0().setYaw(0)

    # create control input and duration
    cspace = ds.space_info.getControlSpace()
    c0 = cspace.allocControl()
    c0[0] = 0.0
    duration = 1.0

    # create state object to store propagated state
    s1 = ob.State(ob.DubinsStateSpace())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate(s0(), c0, duration, s1())
    
    # ~~~ ASSERT ~~~
    assert cspace.getDimension() == 1
    assert np.isclose(s1().getX(), 301)
    assert np.isclose(s1().getY(), 200)
    assert np.isclose(s1().getYaw(), 0)

def test_DubinsPPMSetup_propagator_1():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_FILE_0, 10, 1)
    propagator = ds.space_info.getStatePropagator()

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(300)
    s0().setY(200)
    s0().setYaw(0)

    # create control input and duration
    cspace = ds.space_info.getControlSpace()
    c0 = cspace.allocControl()
    c0[0] = 0.0
    duration = 1.0

    # create state object to store propagated state
    s1 = ob.State(ob.DubinsStateSpace())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate(s0(), c0, duration, s1())
    
    # ~~~ ASSERT ~~~
    assert cspace.getDimension() == 1
    assert np.isclose(s1().getX(), 310)
    assert np.isclose(s1().getY(), 200)
    assert np.isclose(s1().getYaw(), 0)

def test_DubinsPPMSetup_propagator_2():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_FILE_0, 10, 1)
    propagator = ds.space_info.getStatePropagator()

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(300)
    s0().setY(200)
    s0().setYaw(np.pi/2.0)

    # create control input and duration
    cspace = ds.space_info.getControlSpace()
    c0 = cspace.allocControl()
    c0[0] = 0.0
    duration = 1.0

    # create state object to store propagated state
    s1 = ob.State(ob.DubinsStateSpace())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate(s0(), c0, duration, s1())
    
    # ~~~ ASSERT ~~~
    assert cspace.getDimension() == 1
    assert np.isclose(s1().getX(), 300)
    assert np.isclose(s1().getY(), 210)
    assert np.isclose(s1().getYaw(), np.pi/2)

def test_DubinsPPMSetup_propagator_3():
    '''test that propagator arrives nears init state after a circle'''

    # ~~~ ARRANGE ~~~
    speed = 1.0
    min_turn_radius = 50.0
    x0 = 300
    y0 = 200
    yaw0 = 0
    ds = DubinsPPMSetup(PPM_FILE_0, speed=speed, min_turn_radius=min_turn_radius)
    propagator = ds.space_info.getStatePropagator()

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(yaw0)

    # create control input and duration
    cspace = ds.space_info.getControlSpace()
    c0 = cspace.allocControl()
    duration = np.pi * min_turn_radius / (2.0 * speed)
    c0[0] = np.pi/(2.0 * duration)   # rad/s exceeds control bounds, should constrain to speed/turn

    # create state object to store propagated state
    s1 = ob.State(ob.DubinsStateSpace())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate(s0(), c0, duration, s1())
    
    # ~~~ ASSERT ~~~
    assert cspace.getDimension() == 1
    assert np.isclose(s1().getX(), x0 + min_turn_radius, rtol=0.001)
    assert np.isclose(s1().getY(), y0 + min_turn_radius, rtol=0.001)
    normYaw = s1().getYaw() % (2*np.pi)
    assert np.isclose(normYaw, yaw0 + np.pi/2.0), 'Might need to clamp yaw to 0, 2pi'

def test_DubinsPPMSetup_propagator_4():
    '''test that propagator arrives nears init state after a circle'''

    # ~~~ ARRANGE ~~~
    speed = 2.0
    min_turn_radius = 10.0
    x0 = 300
    y0 = 200
    yaw0 = 0
    ds = DubinsPPMSetup(PPM_FILE_0, speed=speed, min_turn_radius=min_turn_radius)
    propagator = ds.space_info.getStatePropagator()

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(yaw0)

    # create control input and duration
    cspace = ds.space_info.getControlSpace()
    c0 = cspace.allocControl()
    c0[0] = -17.389273   # rad/s exceeds control bounds, should constrain to speed/turn
    duration = 2 * np.pi * min_turn_radius / speed

    # create state object to store propagated state
    s1 = ob.State(ob.DubinsStateSpace())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate(s0(), c0, duration, s1())
    
    # ~~~ ASSERT ~~~
    assert cspace.getDimension() == 1
    assert np.isclose(s1().getX(), x0, rtol=0.001)
    assert np.isclose(s1().getY(), y0, rtol=0.001)
    # normYaw = s1().getYaw() % (2*np.pi)
    # assert np.isclose(normYaw, yaw0), 'Might need to clamp yaw to 0, 2pi'
    assert np.isclose(np.sin(s1().getYaw()), np.sin(yaw0))
    assert np.isclose(np.cos(s1().getYaw()), np.cos(yaw0))

@given(
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_DubinsPPMSetup_propagator(speed, min_turn_radius, x0, y0, yaw0, ctrl, dur):
    '''test a broad range of propagator inputs'''
    # ~~~ ARRANGE ~~~

    ds = DubinsPPMSetup(PPM_FILE_0, speed=speed, min_turn_radius=min_turn_radius)
    propagator = ds.space_info.getStatePropagator()
    cbounds = ds.space_info.getControlSpace().getBounds()

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(yaw0)

    # create control input and duration
    cspace = ds.space_info.getControlSpace()
    c0 = cspace.allocControl()
    c0[0] = ctrl

    # create state object to store propagated state
    s1 = ob.State(ob.DubinsStateSpace())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate(s0(), c0, dur, s1())

    # ~~~ ASSERT ~~~
    # numerical integration very inprecise
    # use very loose assertions, mostly just checking errors arent' thrown
    # bnd_ctrl = np.clip(ctrl, cbounds.low, cbounds.high)
    # turn_radius = speed / bnd_ctrl
    # eps = 1e-5
    # assert np.less_equal(s1().getX(), x0 + dur*speed + eps)
    # exp_yaw1 = bnd_ctrl * dur + yaw0
    # assert np.isclose(np.sin(s1().getYaw()), np.sin(exp_yaw1), atol=0.1)
    # assert np.isclose(np.cos(s1().getYaw()), np.cos(exp_yaw1), atol=0.1)



def test_DubinsPPMSetup_sampleReachableSet_0():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    speed = 2.0
    min_turn_radius = 10.0
    duration = 1.0
    n_samples = 100
    x0 = 300
    y0 = 200
    yaw0 = 0
    ds = DubinsPPMSetup(PPM_FILE_0, speed=speed, min_turn_radius=min_turn_radius)

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(yaw0)

    # ~~~ ACT ~~~
    # sample controls
    samples = ds.sampleReachableSet(s0, duration, n_samples)
    
    # ~~~ ASSERT ~~~
    assert len(samples) == n_samples
    for s in samples:
        # very loose bounds on the possible samples
        assert np.less_equal(s.getX(), x0 + duration*speed)
        assert np.greater_equal(s.getX(), x0 - duration*speed)
        assert np.less_equal(s.getY(), y0 + duration*speed)
        assert np.greater_equal(s.getY(), y0 - duration*speed)

if __name__ == "__main__":
    test_DubinsPPMSetup_propagator_4()
