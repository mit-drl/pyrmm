import pickle
import numpy as np

from ompl import base as ob
from ompl import control as oc
from hypothesis import strategies as st
from hypothesis import given

from pyrmm.setups.double_integrator import \
    DoubleIntegrator1DSetup, \
    update_pickler_RealVectorStateSpace2, \
    update_pickler_PathControl_DoubleIntegrator1D

ordered_float_pair_st = st.lists(
    st.floats(allow_infinity=False, allow_nan=False), 
    min_size=2, 
    max_size=2, 
    unique=True).map(lambda x: tuple(sorted(x)))

@given(ordered_float_pair_st)
def test_order_pairs(opair):
    assert opair[0] <= opair[1]

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
    # propagator = ds.space_info.getStatePropagator()

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
        ds.propagate_path(s0, c0, duration, path)
        
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

def test_update_pickler_RealVectorStateSpace2_0():
    """check that RealVector 2 state can be pickled and upickled without change"""
    # ~~~ ARRANGE ~~~
    # set pre-specified random state
    np_state = np.array([0.27323028, 0.19620385])
    omplState = ob.RealVectorStateSpace(2).allocState()
    omplState[0] = np_state[0]
    omplState[1] = np_state[1]

    # ~~~ ACT ~~~
    # update pickler
    update_pickler_RealVectorStateSpace2()

    # pickle and unpicle ompl state
    omplState_copy = pickle.loads(pickle.dumps(omplState))

    # ~~~ ASSERT ~~~
    assert np.isclose(np_state[0], omplState[0])
    assert np.isclose(np_state[1], omplState[1])
    assert np.isclose(omplState[0], omplState_copy[0])
    assert np.isclose(omplState[1], omplState_copy[1])

def test_DoubleIntegrator1DSetup_sampleReachableSet_0():
    """check that sampler produces unique states and controls"""

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

    # define sampler params
    duration = 1.0
    n_samples = 64

    # create state to sample from
    x0 = 0.0
    v0 = 1.0
    s0 = ds.space_info.allocState()
    s0[0] = x0
    s0[1] = v0

    # ~~~ ACT ~~~
    # sample reachable sets (each sample is a trajectory path)
    sampled_paths = ds.sampleReachableSet(s0, duration, n_samples)
    sampled_paths = ds.sampleReachableSet(s0, duration, n_samples)

    # ~~~ ASSERT ~~~
    # check that no repeated control values in sampled paths
    sampled_ctrls = [ds.control_ompl_to_numpy(sp.getControls()[0])[0] for sp in sampled_paths]
    assert not np.any(np.diff(np.sort(sampled_ctrls))==0)

    for sp in sampled_paths:

        # get control and duration of first step in sampled path
        ctrl_dur = sp.getControlDurations()[0]
        ctrl = ds.control_ompl_to_numpy(sp.getControls()[0])

        # create new path object to propagate over first step of sampled path
        sp_sub0 = oc.PathControl(ds.space_info)
        sp_sub0.append(state=ds.space_info.allocState(), control=ds.space_info.allocControl(), duration=0)
        sp_sub0.append(state=ds.space_info.allocState())

        # perform path propagation over first step of sampled trajectory
        ds.propagate_path(
            state = s0,
            control = ctrl,
            duration = ctrl_dur,
            path = sp_sub0
        )

        # ~~~ ASSERT ~~~
        # check that sampled control arrives at first step of sampled trajectory
        assert np.isclose(sp.getState(1)[0], sp_sub0.getState(1)[0])
        assert np.isclose(sp.getState(1)[1], sp_sub0.getState(1)[1])

@given(
    st.floats(min_value=0, max_value=1e3, exclude_min=True, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1e2, exclude_min=True, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1e1, exclude_min=True, allow_nan=False, allow_infinity=False),
    ordered_float_pair_st
)
def test_hypothesis_DoubleIntegrator1DSetup_convert_state_ompl_numpy_0(posb, velb, accb, obst_bounds):
    '''check that conversion between ompl and numpy does not alter state value'''

    # ~~~ ARRANGE ~~~
    pos_bounds = [-posb, posb]
    vel_bounds = [-velb, velb]
    acc_bounds = [-accb, accb]
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # generate random state
    sampler = ds.space_info.allocStateSampler()
    s_ompl_orig = ds.space_info.allocState()
    sampler.sampleUniform(s_ompl_orig)

    # ~~~ ACT ~~~
    # convert to numpy then back to ompl
    s_np = DoubleIntegrator1DSetup.state_ompl_to_numpy(s_ompl_orig)
    s_ompl_copy = ds.space_info.allocState()
    DoubleIntegrator1DSetup.state_numpy_to_ompl(s_np, s_ompl_copy)

    # ~~~ ASSERT ~~~
    # check that original and copy are the same
    assert np.isclose(s_ompl_orig[0], s_ompl_copy[0])
    assert np.isclose(s_ompl_orig[1], s_ompl_copy[1])

    # ~~~ ACT ~~~
    # convert to numpy (in-place) then back to ompl
    s_np_2 = np.zeros(2)
    DoubleIntegrator1DSetup.state_ompl_to_numpy(s_ompl_orig, s_np_2)
    s_ompl_copy_2 = ds.space_info.allocState()
    DoubleIntegrator1DSetup.state_numpy_to_ompl(s_np_2, s_ompl_copy_2)

    # ~~~ ASSERT ~~~
    # check that original and copy are the same
    assert np.isclose(s_ompl_orig[0], s_ompl_copy_2[0])
    assert np.isclose(s_ompl_orig[1], s_ompl_copy_2[1])


@given(
    st.floats(min_value=0, max_value=1e3, exclude_min=True, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1e2, exclude_min=True, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1e1, exclude_min=True, allow_nan=False, allow_infinity=False),
    ordered_float_pair_st
)
def test_hypothesis_DoubleIntegrator1DSetup_convert_control_ompl_numpy_0(posb, velb, accb, obst_bounds):
    '''check that conversion between ompl and numpy does not alter control value'''

    # ~~~ ARRANGE ~~~
    pos_bounds = [-posb, posb]
    vel_bounds = [-velb, velb]
    acc_bounds = [-accb, accb]
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # generate random control
    sampler = ds.space_info.allocControlSampler()
    c_ompl_orig = ds.space_info.allocControl()
    sampler.sample(c_ompl_orig)

    # ~~~ ACT ~~~
    # convert to numpy then back to ompl
    c_np = DoubleIntegrator1DSetup.control_ompl_to_numpy(c_ompl_orig)
    c_ompl_copy = ds.space_info.allocControl()
    DoubleIntegrator1DSetup.control_numpy_to_ompl(c_np, c_ompl_copy)

    # ~~~ ASSERT ~~~
    # check that original and copy are the same
    assert np.isclose(c_ompl_orig[0], c_ompl_copy[0])

    # ~~~ ACT ~~~
    # convert to numpy (in-place) then back to ompl
    c_np_2 = np.zeros(1)
    DoubleIntegrator1DSetup.control_ompl_to_numpy(c_ompl_orig, c_np_2)
    c_ompl_copy_2 = ds.space_info.allocControl()
    DoubleIntegrator1DSetup.control_numpy_to_ompl(c_np_2, c_ompl_copy_2)

    # ~~~ ASSERT ~~~
    # check that original and copy are the same
    assert np.isclose(c_ompl_orig[0], c_ompl_copy_2[0])

@given(
    st.floats(min_value=0, max_value=1e3, exclude_min=True, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1e2, exclude_min=True, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1e1, exclude_min=True, allow_nan=False, allow_infinity=False),
    ordered_float_pair_st,
    st.integers(min_value=2, max_value=128)
)
def test_hypothesis_DoubleIntegrator1DSetup_convert_path_ompl_numpy_0(posb, velb, accb, obst_bounds, nsteps):
    '''check that conversion between ompl and numpy does not alter control value'''

    # ~~~ ARRANGE ~~~
    pos_bounds = [-posb, posb]
    vel_bounds = [-velb, velb]
    acc_bounds = [-accb, accb]
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # generate states, controls and durations 
    np_states_orig = np.random.uniform(
        low=[pos_bounds[0], vel_bounds[0]],
        high=[pos_bounds[1], vel_bounds[1]],
        size=(nsteps, 2))
    np_ctrls_orig = np.random.uniform(*acc_bounds, size=(nsteps-1,1))
    np_durs = np.random.rand(nsteps-1)
    np_times_orig = np.concatenate(([0.0], np.cumsum(np_durs)))

    # instantiate ompl path object
    omplPath = oc.PathControl(ds.space_info)
    for j in range(nsteps-1):
        omplPath.append(state=ds.space_info.allocState(), control=ds.space_info.allocControl(), duration=0)
    omplPath.append(state=ds.space_info.allocState())

    # ~~~ ACT ~~~
    # convert to ompl 
    DoubleIntegrator1DSetup.path_numpy_to_ompl(np_states_orig, np_ctrls_orig, np_times_orig, omplPath)

    # instantiate new np objects to hold converted values
    np_states_copy = np.random.uniform(
        low=[pos_bounds[0], vel_bounds[0]],
        high=[pos_bounds[1], vel_bounds[1]],
        size=(nsteps, 2))
    np_ctrls_copy = np.random.uniform(*acc_bounds, size=(nsteps-1, 1))
    np_times_copy = np.random.rand(nsteps)

    # convert to numpy
    DoubleIntegrator1DSetup.path_ompl_to_numpy(omplPath, np_states_copy, np_ctrls_copy, np_times_copy)

    # ~~~ ASSERT ~~~
    # check that original and copy match
    assert np.allclose(np_states_orig, np_states_copy)
    assert np.allclose(np_ctrls_orig, np_ctrls_copy)
    assert np.allclose(np_times_orig, np_times_copy)


@given(
    st.floats(min_value=0, max_value=1e3, exclude_min=True, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1e2, exclude_min=True, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0, max_value=1e1, exclude_min=True, allow_nan=False, allow_infinity=False),
    ordered_float_pair_st,
    st.integers(min_value=2, max_value=128)
)
def test_hypothesis_update_pickler_PathControl_DoubleIntegrator1D_0(posb, velb, accb, obst_bounds, nsteps):
    """check that path control object can be pickled and unpickled accurately"""

    # ~~~ ARRANGE ~~~
    pos_bounds = [-posb, posb]
    vel_bounds = [-velb, velb]
    acc_bounds = [-accb, accb]
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # generate states, controls and durations 
    np_states_orig = np.random.uniform(
        low=[pos_bounds[0], vel_bounds[0]],
        high=[pos_bounds[1], vel_bounds[1]],
        size=(nsteps, 2))
    np_ctrls_orig = np.random.uniform(*acc_bounds, size=(nsteps-1,1))
    np_durs = np.random.rand(nsteps-1)
    np_times_orig = np.concatenate(([0.0], np.cumsum(np_durs)))

    # instantiate ompl path object
    omplPath = oc.PathControl(ds.space_info)
    for j in range(nsteps-1):
        omplPath.append(state=ds.space_info.allocState(), control=ds.space_info.allocControl(), duration=0)
    omplPath.append(state=ds.space_info.allocState())

    # ~~~ ACT ~~~
    # convert to ompl 
    DoubleIntegrator1DSetup.path_numpy_to_ompl(np_states_orig, np_ctrls_orig, np_times_orig, omplPath)

    # update pickler
    update_pickler_PathControl_DoubleIntegrator1D()

    # pickle and unpicle ompl PathControl object
    omplPath_copy = pickle.loads(pickle.dumps(omplPath))

    # instantiate new np objects to hold converted values
    np_states_copy = np.random.uniform(
        low=[pos_bounds[0], vel_bounds[0]],
        high=[pos_bounds[1], vel_bounds[1]],
        size=(nsteps, 2))
    np_ctrls_copy = np.random.uniform(*acc_bounds, size=(nsteps-1, 1))
    np_times_copy = np.random.rand(nsteps)

    # convert de-pickled ompl object back to numpy
    DoubleIntegrator1DSetup.path_ompl_to_numpy(omplPath_copy, np_states_copy, np_ctrls_copy, np_times_copy)

    # ~~~ ASSERT ~~~
    # check that original and copy match
    assert np.allclose(np_states_orig, np_states_copy)
    assert np.allclose(np_ctrls_orig, np_ctrls_copy)
    assert np.allclose(np_times_orig, np_times_copy)

def test_observation_shape_0():
    """Check that inheritance of observation shape behaves as expected"""

    # ~~~ ARRANGE ~~~
    
    # ~~~ ACT ~~~

    # ~~~ ASSERT ~~~
    assert DoubleIntegrator1DSetup.observation_shape == (3,)



if __name__ == "__main__":
    # test_DoubleIntegrator1DSetup_propagate_path_0()
    # test_DoubleIntegrator1DSetup_estimateRiskMetric_0()
    # test_DoubleIntegrator1DSetup_observeState_0()
    test_DoubleIntegrator1DSetup_sampleReachableSet_0()