import pathlib
import faulthandler
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

    # create state object to store propagated state
    # s1 = ob.State(ob.DubinsStateSpace())

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

def test_DubinsPPMSetup_propagator_1():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_FILE_0, 10, 1)
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

def test_DubinsPPMSetup_propagator_2():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    ds = DubinsPPMSetup(PPM_FILE_0, 10, 1)
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

def test_DubinsPPMSetup_propagator_3():
    '''test that propagator arrives nears init state after a circle'''

    # ~~~ ARRANGE ~~~
    speed = 1.0
    min_turn_radius = 50.0
    x0 = 300
    y0 = 200
    yaw0 = 0
    ds = DubinsPPMSetup(PPM_FILE_0, speed=speed, min_turn_radius=min_turn_radius)
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
    # nsteps = 3
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
    
    # ~~~ ASSERT ~~~
    # assert cspace.getDimension() == 1
    # assert np.isclose(s1().getX(), x0 + min_turn_radius, rtol=0.001)
    # assert np.isclose(s1().getY(), y0 + min_turn_radius, rtol=0.001)
    # normYaw = s1().getYaw() % (2*np.pi)
    # assert np.isclose(normYaw, yaw0 + np.pi/2.0), 'Might need to clamp yaw to 0, 2pi'

def test_DubinsPPMSetup_propagator_4():
    '''test that propagator arrives nears init state after a circle'''

    # ~~~ ARRANGE ~~~
    speed = 2.0
    min_turn_radius = 10.0
    x0 = 300
    y0 = 200
    yaw0 = 0
    ds = DubinsPPMSetup(PPM_FILE_0, speed=speed, min_turn_radius=min_turn_radius)
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
    
    # ~~~ ASSERT ~~~
    # assert cspace.getDimension() == 1
    # assert np.isclose(s1().getX(), x0, rtol=0.001)
    # assert np.isclose(s1().getY(), y0, rtol=0.001)
    # # normYaw = s1().getYaw() % (2*np.pi)
    # # assert np.isclose(normYaw, yaw0), 'Might need to clamp yaw to 0, 2pi'
    # assert np.isclose(np.sin(s1().getYaw()), np.sin(yaw0))
    # assert np.isclose(np.cos(s1().getYaw()), np.cos(yaw0))

@given(
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(allow_nan=False, allow_infinity=False),
    st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.integers(min_value=2, max_value=1e3)
)
def test_hypothesis_DubinsPPMSetup_propagator_error_check(speed, min_turn_radius, x0, y0, yaw0, ctrl, dur, nsteps):
    '''test a broad range of propagator inputs to see if they raise errors'''
    # ~~~ ARRANGE ~~~

    ds = DubinsPPMSetup(PPM_FILE_0, speed=speed, min_turn_radius=min_turn_radius)
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
    # numerical integration very inprecise
    # use very loose assertions, mostly just checking errors arent' thrown
    # bnd_ctrl = np.clip(ctrl, cbounds.low, cbounds.high)
    # turn_radius = speed / bnd_ctrl
    # eps = 1e-5
    # assert np.less_equal(s1().getX(), x0 + dur*speed + eps)
    # exp_yaw1 = bnd_ctrl * dur + yaw0
    # assert np.isclose(np.sin(s1().getYaw()), np.sin(exp_yaw1), atol=0.1)
    # assert np.isclose(np.cos(s1().getYaw()), np.cos(exp_yaw1), atol=0.1)

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
def test_hypothesis_DubinsPPMSetup_propagator_clipped_ctrl(speed, min_turn_radius, x0, y0, yaw0, ctrl_neg, ctrl_dev, dur_scale, nsteps):
    '''test a narrow range of propagator inputs with clipped control value'''
    # ~~~ ARRANGE ~~~

    ds = DubinsPPMSetup(PPM_FILE_0, speed=speed, min_turn_radius=min_turn_radius)
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
    # rndsign = 1 if np.random.rand() > 0.5 else -1
    # rndctrl = 100*np.random.rand()
    ctrl_sign = -1 if ctrl_neg else 1
    ctrl = ctrl_sign * (speed / min_turn_radius + ctrl_dev)
    c0[0] = ctrl

    # create duration to ensure it is less than 2 full revolutions
    bnd_ctrl = np.clip(ctrl, cbounds.low[0], cbounds.high[0])
    dur = (4*np.pi / bnd_ctrl) * dur_scale

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
    assert np.isclose(path.getState(nsteps-1).getX(), exp_x1, rtol=1e-5)
    assert np.isclose(path.getState(nsteps-1).getY(), exp_y1, rtol=1e-5)
    assert np.isclose(np.sin(path.getState(nsteps-1).getYaw()), np.sin(exp_yaw1))
    assert np.isclose(np.cos(path.getState(nsteps-1).getYaw()), np.cos(exp_yaw1))
    bnd_ctrl_obj = cspace.allocControl()
    bnd_ctrl_obj[0] = bnd_ctrl
    for i in range(nsteps-1):
        assert np.isclose(path.getControlDuration(i), dur/(nsteps-1))
        assert si.getControlSpace().equalControls(path.getControl(i), bnd_ctrl_obj)

    # bnd_ctrl = np.clip(ctrl, cbounds.low, cbounds.high)
    # turn_radius = speed / bnd_ctrl
    # eps = 1e-5
    # assert np.less_equal(s1().getX(), x0 + dur*speed + eps)
    # exp_yaw1 = bnd_ctrl * dur + yaw0
    # assert np.isclose(np.sin(s1().getYaw()), np.sin(exp_yaw1))
    # assert np.isclose(np.cos(s1().getYaw()), np.cos(exp_yaw1))
    # exp_x1 = x0 + exp_turn_radius * ctrl_sign * (np.sin(exp_yaw1) - np.sin(yaw0))
    # exp_y1 = y0 + exp_turn_radius * ctrl_sign * (-np.cos(exp_yaw1) + np.cos(yaw0))
    # assert np.isclose(s1().getX(), exp_x1)
    # assert np.isclose(s1().getY(), exp_y1)


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
    ds = DubinsPPMSetup(PPM_FILE_0, speed=speed, min_turn_radius=min_turn_radius)

    # create initial state
    s0 = ob.State(ob.DubinsStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(yaw0)

    # ~~~ ACT ~~~
    # sample controls
    samples = ds.sampleReachableSet(s0, duration, n_samples, n_steps=n_steps)
    
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

if __name__ == "__main__":
    faulthandler.enable()
    # test_DubinsPPMSetup_propagator_3()
    # test_hypothesis_DubinsPPMSetup_propagator_clipped_ctrl()
    test_DubinsPPMSetup_sampleReachableSet_0()
    pass
