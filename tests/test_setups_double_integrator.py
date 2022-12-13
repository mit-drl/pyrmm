import numpy as np

from ompl import base as ob
from ompl import control as oc
from hypothesis import strategies as st
from hypothesis import given

from pyrmm.setups.double_integrator import DoubleIntegrator1DSetup


def test_DoubleIntegrator1DSetup_estimateRiskMetric_0():
    '''Check risk metric for known, fixed accel, deterministic sytem'''
    
    # ~~~ ARRANGE ~~~
    pos_bounds = [-8, 8]
    vel_bounds = [-2, 2]
    acc_bounds = [0, 0] # deterministic system with no accel
    obst_bounds = [1.0, 1.00001]    # very slim obstacle
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)
    s0 = ds.space_info.allocState()
    s0[0] = 0.0 # starts at origin
    s0[1] = 1.0 # start with init positive velocity [m/s]

    # distance and depth does not reach boundary
    r, _, _ = ds.estimateRiskMetric(
            state=s0,
            trajectory=None,
            distance=0.2,
            branch_fact=8,
            depth=4,
            n_steps=2
    )
    assert np.isclose(0.0, r)

    # distance and depth barely not reach boundary
    r, _, _ = ds.estimateRiskMetric(
            state=s0,
            trajectory=None,
            distance=0.25 - 1e-6,
            branch_fact=8,
            depth=4,
            n_steps=2
    )
    assert np.isclose(0.0, r)

    # distance and depth barely passes boundary
    r, _, _ = ds.estimateRiskMetric(
            state=s0,
            trajectory=None,
            distance=0.25 + 1e-6,
            branch_fact=8,
            depth=4,
            n_steps=2
    )
    assert np.isclose(1.0, r)

    # distance and depth completely passes whole obstacle 
    r, _, _ = ds.estimateRiskMetric(
            state=s0,
            trajectory=None,
            distance=0.6,
            branch_fact=8,
            depth=4,
            n_steps=2
    )
    assert np.isclose(1.0, r)

def test_DoubleIntegrator1DSetup_propagate_path_0():
    '''test that propagator arrives at expected state'''

    # ~~~ ARRANGE ~~~
    pos_bounds = [-8, 8]
    vel_bounds = [-2, 2]
    acc_bounds = [-1, 1]
    obst_bounds = [10, 11]
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)
    si = ds.space_info
    propagator = ds.space_info.getStatePropagator()

    # set cases to be tested
    init_pos = [0.0, 0.0, 0.0, 10, 0.0, -1.0]
    init_vel = [0.0, 1.0, 0.0, 3.0, 1.0, -1.0]
    acc_inpt = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0]
    ctrl_dur = [1.0, 1.0, 1.0, 1.0, 2.0, 1.0]
    end_pos = [0.0, 1.0, 0.5, 13.5, 0.0, -2.5]
    end_vel = [0.0, 1.0, 1.0, 4.0, -1.0, -2.0]

    for i in range(len(init_pos)):

        # create initial state
        s0 = ds.space_info.allocState()
        s0[0] = init_pos[i]
        s0[1] = init_vel[i]

        # create control input and duration
        cspace = ds.space_info.getControlSpace()
        c0 = cspace.allocControl()
        c0[0] = acc_inpt[i]
        duration = ctrl_dur[i]

        # create path object and alloc 2 states
        path = oc.PathControl(si)
        path.append(state=si.allocState(), control=si.allocControl(), duration=0)
        path.append(state=si.allocState())

        # ~~~ ACT ~~~
        # propagate state
        propagator.propagate_path(s0, c0, duration, path)
        
        # ~~~ ASSERT ~~~
        assert cspace.getDimension() == 1
        assert path.getStateCount() == 2
        assert path.getControlCount() == 1
        assert np.isclose(path.getState(0)[0], init_pos[i])
        assert np.isclose(path.getState(0)[1], init_vel[i])
        assert np.isclose(path.getControl(0)[0], acc_inpt[i])
        assert np.isclose(path.getControlDuration(0), ctrl_dur[i])
        assert np.isclose(path.getState(1)[0], end_pos[i], rtol=1e-7, atol=1e-7)
        assert np.isclose(path.getState(1)[1], end_vel[i])

def test_DoubleIntegrator1DSetup_observeState_0():
    '''test observation is as expected'''

    # ~~~ ARRANGE ~~~
    pos_bounds = [-8, 8]
    vel_bounds = [-2, 2]
    acc_bounds = [-1, 1]
    obst_bounds = [1, 2]
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # set cases to be tested
    pos = [0.0, 3.0, 1.0, 2.0, -8.0, 8.0]
    vel = [0.0, 0.0, -1.0, 1.0, 2.0, 2.0]
    exp_obsv = [
        [1000.0, 1.0, 0.0],
        [1.0, 1000.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
        [1000.0, 9.0, 2.0],
        [6.0, 1000.0, 2.0]
    ]

    for i in range(len(pos)):

        # create initial state
        s0 = ds.space_info.allocState()
        s0[0] = pos[i]
        s0[1] = vel[i]


        # ~~~ ACT ~~~
        # take observation
        obsv = ds.observeState(s0)
        
        # ~~~ ASSERT ~~~
        assert np.allclose(obsv, exp_obsv[i])


if __name__ == "__main__":
    # test_DoubleIntegrator1DSetup_propagate_path_0()
    # test_DoubleIntegrator1DSetup_estimateRiskMetric_0()
    test_DoubleIntegrator1DSetup_observeState_0()