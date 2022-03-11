import pathlib
import faulthandler
import numpy as np

from ompl import base as ob
from ompl import control as oc
from hypothesis import strategies as st
from hypothesis import given
from pyrmm.setups.dubins import DubinsPPMSetup


PPM_BORDER_FILE = str(pathlib.Path(__file__).parent.absolute().joinpath("border_640x400.ppm"))
PPM_PARTITION_FILE = str(pathlib.Path(__file__).parent.absolute().joinpath("partition_640x400.ppm"))

def test_DubinsPPMSetup_init_0():
    '''test that DubinsPPMSetup constructed without error'''
    DubinsPPMSetup(PPM_BORDER_FILE, 1, 1)

def test_DubinsPPMSetup_state_checker_0():
    '''test that known states validity'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_BORDER_FILE, 1, 1)
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

def test_DubinsPPMSetup_propagate_path_0():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_BORDER_FILE, 1, 1)
    si = ds.space_info
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

    # create path object and alloc 2 states
    path = oc.PathControl(si)
    path.append(state=si.allocState(), control=si.allocControl(), duration=0)
    path.append(state=si.allocState())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate_path(s0(), c0, duration, path)
    
    # ~~~ ASSERT ~~~
    assert cspace.getDimension() == 1
    assert path.getStateCount() == 2
    assert path.getControlCount() == 1
    assert np.isclose(path.getState(0).getX(), 300)
    assert np.isclose(path.getState(0).getY(), 200)
    assert np.isclose(path.getState(0).getYaw(), 0)
    assert np.isclose(path.getControl(0)[0], 0.0)
    assert np.isclose(path.getControlDuration(0), 1.0)
    assert np.isclose(path.getState(1).getX(), 301)
    assert np.isclose(path.getState(1).getY(), 200)
    assert np.isclose(path.getState(1).getYaw(), 0)

def test_DubinsPPMSetup_propagate_path_1():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_BORDER_FILE, 10, 1)
    si = ds.space_info
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

    # create path object and alloc 2 states
    path = oc.PathControl(si)
    path.append(state=si.allocState(), control=si.allocControl(), duration=0)
    path.append(state=si.allocState())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate_path(s0(), c0, duration, path)
    
    # ~~~ ASSERT ~~~
    assert cspace.getDimension() == 1
    assert path.getStateCount() == 2
    assert path.getControlCount() == 1
    assert np.isclose(path.getState(0).getX(), 300)
    assert np.isclose(path.getState(0).getY(), 200)
    assert np.isclose(path.getState(0).getYaw(), 0)
    assert np.isclose(path.getControl(0)[0], 0.0)
    assert np.isclose(path.getControlDuration(0), 1.0)
    assert np.isclose(path.getState(1).getX(), 310)
    assert np.isclose(path.getState(1).getY(), 200)
    assert np.isclose(path.getState(1).getYaw(), 0)

def test_DubinsPPMSetup_propagate_path_2():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_BORDER_FILE, 10, 1)
    si = ds.space_info
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

    # create path object and alloc 2 states
    path = oc.PathControl(si)
    path.append(state=si.allocState(), control=si.allocControl(), duration=0)
    path.append(state=si.allocState())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate_path(s0(), c0, duration, path)
    
    # ~~~ ASSERT ~~~
    assert cspace.getDimension() == 1
    assert path.getStateCount() == 2
    assert path.getControlCount() == 1
    assert np.isclose(path.getState(0).getX(), 300)
    assert np.isclose(path.getState(0).getY(), 200)
    assert np.isclose(path.getState(0).getYaw(), np.pi/2)
    assert np.isclose(path.getControl(0)[0], 0.0)
    assert np.isclose(path.getControlDuration(0), 1.0)
    assert np.isclose(path.getState(1).getX(), 300)
    assert np.isclose(path.getState(1).getY(), 210)
    assert np.isclose(path.getState(1).getYaw(), np.pi/2)

def test_DubinsPPMSetup_propagate_path_3():
    '''test that propagator arrives nears init state after a circle'''

    # ~~~ ARRANGE ~~~
    speed = 1.0
    min_turn_radius = 50.0
    x0 = 300
    y0 = 200
    yaw0 = 0
    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)
    si = ds.space_info
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

    # create path object and alloc a randomized number of intermediate steps
    path = oc.PathControl(si)
    nsteps = np.random.randint(5, 100)
    for _ in range(nsteps-1):
        path.append(state=si.allocState(), control=si.allocControl(), duration=0)
    path.append(state=si.allocState())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate_path(s0(), c0, duration, path)
    
    # ~~~ ASSERT ~~~
    assert cspace.getDimension() == 1
    assert path.getStateCount() == nsteps
    assert path.getControlCount() == nsteps - 1
    assert np.isclose(path.getState(0).getX(), 300)
    assert np.isclose(path.getState(0).getY(), 200)
    assert np.isclose(path.getState(0).getYaw(), 0)
    assert np.isclose(path.getState(nsteps-1).getX(), x0 + min_turn_radius, rtol=0.001)
    assert np.isclose(path.getState(nsteps-1).getY(), y0 + min_turn_radius, rtol=0.001)
    normYaw = path.getState(nsteps-1).getYaw() % (2*np.pi)
    assert np.isclose(normYaw, yaw0 + np.pi/2.0), 'Might need to clamp yaw to 0, 2pi'
    for i in range(nsteps-1):
        assert np.isclose(path.getControlDuration(i), duration/(nsteps-1))
        assert si.getControlSpace().equalControls(path.getControl(i), c0)

def test_DubinsPPMSetup_propagate_path_4():
    '''test that propagator arrives nears init state after a circle'''

    # ~~~ ARRANGE ~~~
    speed = 2.0
    min_turn_radius = 10.0
    x0 = 300
    y0 = 200
    yaw0 = 0
    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)
    si = ds.space_info
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

    # create path object and alloc a randomized number of intermediate steps
    path = oc.PathControl(si)
    nsteps = np.random.randint(5, 100)
    # nsteps = 3
    for _ in range(nsteps-1):
        path.append(state=si.allocState(), control=si.allocControl(), duration=0)
    path.append(state=si.allocState())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate_path(s0(), c0, duration, path)
    
    # ~~~ ASSERT ~~~
    bnd_ctrl = cspace.allocControl()
    bnd_ctrl[0] = -speed/min_turn_radius
    assert cspace.getDimension() == 1
    assert path.getStateCount() == nsteps
    assert path.getControlCount() == nsteps - 1
    assert np.isclose(path.getState(0).getX(), x0)
    assert np.isclose(path.getState(0).getY(), y0)
    assert np.isclose(path.getState(0).getYaw(), yaw0)
    assert np.isclose(path.getState(nsteps-1).getX(), x0, rtol=0.001)
    assert np.isclose(path.getState(nsteps-1).getY(), y0, rtol=0.001)
    assert np.isclose(np.sin(path.getState(nsteps-1).getYaw()), np.sin(yaw0))
    assert np.isclose(np.cos(path.getState(nsteps-1).getYaw()), np.cos(yaw0))
    for i in range(nsteps-1):
        assert np.isclose(path.getControlDuration(i), duration/(nsteps-1))
        assert si.getControlSpace().equalControls(path.getControl(i), bnd_ctrl)

@given(
    st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.integers(min_value=2, max_value=1e2)
)
def test_hypothesis_DubinsPPMSetup_propagate_path_error_check(speed, min_turn_radius, x0, y0, yaw0, ctrl, dur, nsteps):
    '''test a broad range of propagator inputs to see if they raise errors'''
    # ~~~ ARRANGE ~~~

    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)
    si = ds.space_info
    propagator = ds.space_info.getStatePropagator()

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(yaw0)

    # create control input and duration
    cspace = ds.space_info.getControlSpace()
    c0 = cspace.allocControl()
    c0[0] = ctrl

    # create path object and alloc a randomized number of intermediate steps
    path = oc.PathControl(si)
    for _ in range(nsteps-1):
        path.append(state=si.allocState(), control=si.allocControl(), duration=0)
    path.append(state=si.allocState())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate_path(s0(), c0, dur, path)

    # ~~~ ASSERT ~~~
    pass

@given(
    st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-3, max_value=1, allow_nan=False, allow_infinity=False),
    st.integers(min_value=2, max_value=1e2)
)
def test_hypothesis_DubinsPPMSetup_propagate_path_clipped_ctrl(speed, min_turn_radius, x0, y0, yaw0, ctrl_neg, ctrl_dev, dur_scale, nsteps):
    '''test a narrow range of propagator inputs with clipped control value'''
    # ~~~ ARRANGE ~~~

    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)
    si = ds.space_info
    propagator = ds.space_info.getStatePropagator()
    cbounds = ds.space_info.getControlSpace().getBounds()

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(yaw0)

    # create control input to ensure it exceeds cspace bounds
    cspace = ds.space_info.getControlSpace()
    c0 = cspace.allocControl()
    ctrl_sign = -1 if ctrl_neg else 1
    ctrl = ctrl_sign * (speed / min_turn_radius + ctrl_dev)
    c0[0] = ctrl

    # create duration to ensure it is less than 2 full revolutions
    bnd_ctrl = np.clip(ctrl, cbounds.low[0], cbounds.high[0])
    dur = abs(4*np.pi / bnd_ctrl) * dur_scale

    # create path object and alloc a randomized number of intermediate steps
    path = oc.PathControl(si)
    for _ in range(nsteps-1):
        path.append(state=si.allocState(), control=si.allocControl(), duration=0)
    path.append(state=si.allocState())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate_path(s0(), c0, dur, path)

    # ~~~ ASSERT ~~~
    # numerical integration very inprecise
    # use very loose assertions, mostly just checking errors arent' thrown
    assert cspace.getDimension() == 1
    assert path.getStateCount() == nsteps
    assert path.getControlCount() == nsteps - 1
    exp_turn_radius = abs(speed / bnd_ctrl)
    assert np.isclose(exp_turn_radius, min_turn_radius)
    assert np.isclose(path.getState(0).getX(), x0)
    assert np.isclose(path.getState(0).getY(), y0)
    assert np.isclose(path.getState(0).getYaw(), yaw0)
    exp_yaw1 = bnd_ctrl * dur + yaw0
    exp_x1 = x0 + exp_turn_radius * ctrl_sign * (np.sin(exp_yaw1) - np.sin(yaw0))
    exp_y1 = y0 + exp_turn_radius * ctrl_sign * (-np.cos(exp_yaw1) + np.cos(yaw0))
    assert np.isclose(path.getState(nsteps-1).getX(), exp_x1, rtol=1e-4)
    assert np.isclose(path.getState(nsteps-1).getY(), exp_y1, rtol=1e-4)
    assert np.isclose(np.sin(path.getState(nsteps-1).getYaw()), np.sin(exp_yaw1))
    assert np.isclose(np.cos(path.getState(nsteps-1).getYaw()), np.cos(exp_yaw1))
    bnd_ctrl_obj = cspace.allocControl()
    bnd_ctrl_obj[0] = bnd_ctrl
    for i in range(nsteps-1):
        assert np.isclose(path.getControlDuration(i), dur/(nsteps-1))
        assert si.getControlSpace().equalControls(path.getControl(i), bnd_ctrl_obj)


def test_DubinsPPMSetup_propagator_0():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_BORDER_FILE, 1, 1)
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
    ds = DubinsPPMSetup(PPM_BORDER_FILE, 10, 1)
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
    ds = DubinsPPMSetup(PPM_BORDER_FILE, 10, 1)
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
    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)
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
    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)
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
def test_hypothesis_DubinsPPMSetup_propagator_error_check(speed, min_turn_radius, x0, y0, yaw0, ctrl, dur):
    '''test a broad range of propagator inputs to see if they raise errors'''
    # ~~~ ARRANGE ~~~

    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)
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
    pass

@given(
    st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-3, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_DubinsPPMSetup_propagator_clipped_ctrl(speed, min_turn_radius, x0, y0, yaw0, ctrl_neg, ctrl_dev, dur_scale):
    '''test a narrow range of propagator inputs with clipped control value'''
    # ~~~ ARRANGE ~~~

    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)
    propagator = ds.space_info.getStatePropagator()
    cbounds = ds.space_info.getControlSpace().getBounds()

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(yaw0)

    # create control input to ensure it exceeds cspace bounds
    cspace = ds.space_info.getControlSpace()
    c0 = cspace.allocControl()
    ctrl_sign = -1 if ctrl_neg else 1
    ctrl = ctrl_sign * (speed / min_turn_radius + ctrl_dev)
    c0[0] = ctrl

    # create duration to ensure it is less than 2 full revolutions
    bnd_ctrl = np.clip(ctrl, cbounds.low[0], cbounds.high[0])
    dur = abs(4*np.pi / bnd_ctrl) * dur_scale

    # create state object to store propagated state
    s1 = ob.State(ob.DubinsStateSpace())

    # ~~~ ACT ~~~
    # propagate state
    propagator.propagate(s0(), c0, dur, s1())

    # ~~~ ASSERT ~~~
    # numerical integration very inprecise
    # use very loose assertions, mostly just checking errors arent' thrown
    exp_turn_radius = abs(speed / bnd_ctrl)
    assert np.isclose(exp_turn_radius, min_turn_radius)
    exp_yaw1 = bnd_ctrl * dur + yaw0
    assert np.isclose(np.sin(s1().getYaw()), np.sin(exp_yaw1))
    assert np.isclose(np.cos(s1().getYaw()), np.cos(exp_yaw1))
    exp_x1 = x0 + exp_turn_radius * ctrl_sign * (np.sin(exp_yaw1) - np.sin(yaw0))
    exp_y1 = y0 + exp_turn_radius * ctrl_sign * (-np.cos(exp_yaw1) + np.cos(yaw0))
    assert np.isclose(s1().getX(), exp_x1, rtol=1e-3)
    assert np.isclose(s1().getY(), exp_y1, rtol=1e-3)

def test_DubinsPPMSetup_sampleReachableSet_0():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    speed = 2.0
    min_turn_radius = 10.0
    duration = 1.0
    n_samples = 1000
    n_steps = 10
    x0 = 300
    y0 = 200
    yaw0 = 0
    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(yaw0)

    # ~~~ ACT ~~~
    # sample controls
    samples = ds.sampleReachableSet(s0(), duration, n_samples, n_steps=n_steps)
    
    # ~~~ ASSERT ~~~
    assert len(samples) == n_samples
    for pth in samples:
        # very loose bounds on the possible samples
        assert np.isclose(pth.getState(0).getX(), x0)
        assert np.isclose(pth.getState(0).getY(), y0)
        assert np.isclose(pth.getState(0).getYaw(), yaw0)
        assert np.less_equal(pth.getState(n_steps-1).getX(), x0 + duration*speed)
        assert np.greater_equal(pth.getState(n_steps-1).getX(), x0 - duration*speed)
        assert np.less_equal(pth.getState(n_steps-1).getY(), y0 + duration*speed)
        assert np.greater_equal(pth.getState(n_steps-1).getY(), y0 - duration*speed)

def test_DubinsPPMSetup_estimateRiskMetric_zero_risk_region_0():
    '''check that, when sampled away from obstacles, always produces zero risk'''

    # ~~~ ARRANGE ~~~
    speed = 10.0
    min_turn_radius = 50.0
    n_samples = 25
    duration = 2.0
    branch_fact = 32
    tree_depth = 2
    n_steps = 2
    x0 = 320
    y0 = 200
    yaw0 = 0
    near_dist = 150

    # create system setup
    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)

    # create sampling point
    s_near = ds.space_info.allocState()
    s_near.setX(x0)
    s_near.setY(y0)
    s_near.setYaw(yaw0)

    # create sampler
    sampler = ds.space_info.allocStateSampler()
    ssamples = n_samples * [None] 
    rmetrics = n_samples * [None]

    # ~~~ ACT ~~~
    # sample states in zero-risk region and compute risk metrics

    for i in range(n_samples):

        # assign state and sample
        ssamples[i] = ds.space_info.allocState()
        sampler.sampleUniformNear(ssamples[i], s_near, near_dist)

        # compute risk metric
        rmetrics[i] = ds.estimateRiskMetric(ssamples[i], None, duration, branch_fact, tree_depth, n_steps)

        # print("Debug: risk metric={} at state ({},{},{})".format(rmetrics[i], ssamples[i].getX(), ssamples[i].getY(), ssamples[i].getYaw()))

        assert np.isclose(rmetrics[i], 0.0), "non-zero risk metric of {} for state ({},{},{})".format(rmetrics[i], ssamples[i].getX(), ssamples[i].getY(), ssamples[i].getYaw())

def test_DubinsPPMSetup_cast_ray_0():
    '''check that a rays cast from a known state produce expected lengths'''

    # ~~~ ARRANGE ~~~
    speed = 10.0
    min_turn_radius = 50.0
    x0 = 320
    y0 = 200
    yaw0 = 0

    # create system setup
    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=speed, min_turn_radius=min_turn_radius)

    # create sampling point
    s = ds.space_info.allocState()
    s.setX(x0)
    s.setY(y0)
    s.setYaw(yaw0)

    # ~~~ ACT ~~~
    # cast forward ray
    l = ds.cast_ray(s, 0.0, 1.0)

    # ~~~ ASSERT ~~~
    # first black pixel is at 615, therefore expected ray length = 615-320=295
    assert np.isclose(l, 295)

@given(
    # st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False),
    # st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-3, max_value=100, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_DubinsPPMSetup_cast_ray(x0, y0, yaw0, theta, res):
    '''check a range of rays cast for expected lengths'''

    # ~~~ ARRANGE ~~~

    # create system setup
    ds = DubinsPPMSetup(PPM_BORDER_FILE, speed=1.0, min_turn_radius=1.0)

    # create sampling point
    s = ds.space_info.allocState()
    s.setX(x0)
    s.setY(y0)
    s.setYaw(yaw0)

    # specify known obstacles
    xlow = 25
    xhigh = 615
    ylow = 25
    yhigh = 375

    # ~~~ ACT ~~~
    # cast forward ray
    l = ds.cast_ray(s, theta, res)

    # compute expected ray length
    ray_head = yaw0 + theta
    ray_vec = np.array([np.cos(ray_head), np.sin(ray_head)])
    x_vec = np.array([1,0])
    y_vec = np.array([0,1])
    th_x = np.arccos(np.dot(x_vec, ray_vec))
    th_y = np.arccos(np.dot(y_vec, ray_vec))

    if x0 <= xlow or x0 >= xhigh or y0 <= ylow or y0 >= yhigh:
        exp_l = 0.0

    else:
        if th_x < np.pi/2:
            # intersects xhigh
            exp_l_x = (xhigh-x0)/np.cos(th_x)

        elif th_x > np.pi/2:
            # intersects xlow
            exp_l_x = (x0-xlow)/np.cos(np.pi-th_x)
        else:
            exp_l_x = np.inf

        if th_y < np.pi/2:
            # intersect yhigh
            exp_l_y = (yhigh-y0)/np.cos(th_y)

        elif th_y > np.pi/2:
            # intersect ylow
            exp_l_y = (y0-ylow)/np.cos(np.pi-th_y)

        else:
            exp_l_y = np.inf

        exp_l = np.min([exp_l_x, exp_l_y])


    # ~~~ ASSERT ~~~
    assert np.greater_equal(l, 0.0)
    assert np.greater_equal(l, exp_l-res)
    assert np.less_equal(l, exp_l+res)

def test_DubinsPPMSetup_isPathValid_0():
    '''basic check that path through thin obstacle is invalid'''

    # ~~~ ARRANGE ~~~

    x0 = 100
    y0 = 200
    yaw0 = 0
    x1 = 540
    y1 = 200
    yaw1 = 0

    # create system setup
    ds = DubinsPPMSetup(PPM_PARTITION_FILE, speed=1.0, min_turn_radius=1.0)

    # create states for path
    s0 = ds.space_info.allocState()
    s0.setX(x0)
    s0.setY(y0)
    s0.setYaw(yaw0)

    s1 = ds.space_info.allocState()
    s1.setX(x1)
    s1.setY(y1)
    s1.setYaw(yaw1)

    # create a 2-step path that crosses the parition obstacle
    pth = oc.PathControl(ds.space_info)
    pth.append(s0)
    pth.append(s1)

    # ~~~ ACT ~~~
    # check path is valid
    is_valid = ds.isPathValid(pth)

    # ~~~ ASSERT ~~~
    assert not is_valid

@given(
    st.floats(min_value=0, max_value=640, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=0, max_value=400, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=640, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=0, max_value=400, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_DubinsPPMSetup_isPathValid_2point(x0, y0, yaw0, x1, y1, yaw1):
    '''randomly generate 2-point path and check path collision with partition'''
    # ~~~ ARRANGE ~~~

    # create system setup
    ds = DubinsPPMSetup(PPM_PARTITION_FILE, speed=1.0, min_turn_radius=1.0)

    # create states for path
    s0 = ds.space_info.allocState()
    s0.setX(x0)
    s0.setY(y0)
    s0.setYaw(yaw0)

    s1 = ds.space_info.allocState()
    s1.setX(x1)
    s1.setY(y1)
    s1.setYaw(yaw1)

    # create a 2-step path that crosses the parition obstacle
    pth = oc.PathControl(ds.space_info)
    pth.append(s0)
    pth.append(s1)

    # ~~~ ACT ~~~
    # check path is valid
    is_valid = ds.isPathValid(pth)

    # ~~~ ASSERT ~~~
    if (x0 < 320 and x1 < 320) or (x0 >= 321 and x1 >= 321):
        # both points to left or right of partition
        assert is_valid
    else:
        # points cross partion
        assert not is_valid

@given(
    st.floats(min_value=0, max_value=640, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=0, max_value=400, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=640, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=0, max_value=400, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=640, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=0, max_value=400, allow_nan=False, allow_infinity=False, exclude_min=True, exclude_max=True),
    st.floats(min_value=-10*np.pi, max_value=10*np.pi, allow_nan=False, allow_infinity=False),
)
def test_hypothesis_DubinsPPMSetup_isPathValid_3point(x0, y0, yaw0, x1, y1, yaw1, x2, y2, yaw2):
    '''randomly generate 3-point path and check path collision with partition'''
    # ~~~ ARRANGE ~~~

    # create system setup
    ds = DubinsPPMSetup(PPM_PARTITION_FILE, speed=1.0, min_turn_radius=1.0)

    # create states for path
    s0 = ds.space_info.allocState()
    s0.setX(x0)
    s0.setY(y0)
    s0.setYaw(yaw0)

    s1 = ds.space_info.allocState()
    s1.setX(x1)
    s1.setY(y1)
    s1.setYaw(yaw1)

    s2 = ds.space_info.allocState()
    s2.setX(x2)
    s2.setY(y2)
    s2.setYaw(yaw2)

    # create a 2-step path that crosses the parition obstacle
    pth = oc.PathControl(ds.space_info)
    pth.append(s0)
    pth.append(s1)
    pth.append(s2)

    # ~~~ ACT ~~~
    # check path is valid
    is_valid = ds.isPathValid(pth)

    # ~~~ ASSERT ~~~
    if (x0 < 320 and x1 < 320 and x2 < 320) or (x0 >= 321 and x1 >= 321 and x2 >= 321):
        # both points to left or right of partition
        assert is_valid
    else:
        # points cross partion
        assert not is_valid


if __name__ == "__main__":
    faulthandler.enable()
    test_DubinsPPMSetup_cast_ray_0()
    # test_hypothesis_DubinsPPMSetup_cast_ray(320, 200, 0, 0, 1)
    # test_DubinsPPMSetup_propagator_3()
    # test_hypothesis_DubinsPPMSetup_propagator_clipped_ctrl()
    # test_DubinsPPMSetup_sampleReachableSet_0()
    test_DubinsPPMSetup_state_checker_0()
    test_DubinsPPMSetup_estimateRiskMetric_zero_risk_region_0()
    pass
