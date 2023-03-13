import multiprocess
import numpy as np
from functools import partial
from types import SimpleNamespace

from pyrmm.datagen.sampler import sample_risk_metrics
from pyrmm.setups.double_integrator import \
    DoubleIntegrator1DSetup, \
    update_pickler_RealVectorStateSpace2, \
    update_pickler_PathControl_DoubleIntegrator1D

def get_non_unique_indices(arr):

    # get unique values and their indices
    _, unique_inds = np.unique(arr, return_index=True)

    # use setdiff1d to find values not in unique indices
    non_unique_inds = np.setdiff1d(
        np.arange(arr.size),
        unique_inds, 
        assume_unique=True)

    return non_unique_inds

def get_non_unique_data(arr):

    non_unique_inds = get_non_unique_indices(arr)

    # get values at non-unique indices
    non_unique_vals = arr[non_unique_inds]

    # get UNIQUE values of non-unique values 
    # (kinda confusing, but makes sense if you think about it
    # from the set of all elements that are unique in the original arr,
    # find those that are unique to that set)
    unique_non_unique_vals = np.unique(non_unique_vals)

    non_unique_data = []
    for unuv in unique_non_unique_vals:
        unuv_inds = np.where(np.isclose(arr, unuv))
        non_unique_data.append((unuv, unuv_inds))

    return non_unique_data


def test_multiprocess_sampleReachableSet_0():
    """Check that states that come from reachable set sampling in multiprocess setting are all different"""

    # ~~~ ARRANGE ~~~
    # create double integrator object
    pos_bounds = [-32, 32]
    vel_bounds = [-8, 8]
    acc_bounds = [-4, 4] # deterministic system with no accel
    obst_bounds = [1.0, 1.00001]    # very slim obstacle
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # update pickler to allow parallelization of ompl objects
    update_pickler_RealVectorStateSpace2()
    update_pickler_PathControl_DoubleIntegrator1D()

    # create multiprocess task pool
    n_cores = 6
    n_top_samples_per_iter = 8
    n_sub_samples = 4
    n_steps_per_path = 5
    n_iters = 16
    distance = 2.0
    for iter in range(n_iters):
        # pool = multiprocess.Pool(n_cores, maxtasksperchild=32)

        with multiprocess.Pool(n_cores, maxtasksperchild=1) as pool:
            # create set of top-level state samples for this iteration
            sampler = ds.space_info.allocStateSampler()
            states = n_top_samples_per_iter * [None] 
            np_paths_states = n_top_samples_per_iter * [None] 
            np_paths_ctrls = n_top_samples_per_iter * [None] 
            np_paths_times = n_top_samples_per_iter * [None] 
            for i in range(n_top_samples_per_iter):

                # assign state
                states[i] = ds.space_info.allocState()

                # sample only valid states
                while True:
                    sampler.sampleUniform(states[i])
                    if ds.space_info.isValid(states[i]):
                        break

                # instantiate numpy trajectory objects
                np_paths_states[i] = [np.empty((n_steps_per_path,2)) for j in range(n_sub_samples)]
                np_paths_ctrls[i] = [np.empty((n_steps_per_path-1,1)) for j in range(n_sub_samples)]
                np_paths_times[i] = [np.empty(n_steps_per_path) for j in range(n_sub_samples)]

            # create partial function
            sampleReachableSet_partial = partial(
                ds.sampleReachableSet,
                distance = distance, 
                n_samples = n_sub_samples,
                n_steps = n_steps_per_path
            )

            # ~~~ ACT ~~~
            # call reachable set sampler
            reachset_samples_iter = pool.imap(sampleReachableSet_partial, states)

            # track multiprocess progress
            reachset_samples = []
            for i in range(n_top_samples_per_iter):
                reachset_samples.append(reachset_samples_iter.next())

            pool.close()
            pool.join()
        del pool

        # Extract states for comparison
        for i in range(n_top_samples_per_iter):
            for j in range(n_sub_samples):
                DoubleIntegrator1DSetup.path_ompl_to_numpy(
                    reachset_samples[i][j], 
                    np_paths_states[i][j],
                    np_paths_ctrls[i][j],
                    np_paths_times[i][j],
                    DoubleIntegrator1DSetup._state_ompl_to_numpy,
                    DoubleIntegrator1DSetup._control_ompl_to_numpy)

                # within a sub-sample, assert all steps have same control inputs
                assert len(np.unique(np_paths_ctrls[i][j])) == 1

            # within a top-level sample, assert that all initial states match
            pos0_i = np.array([s_i[0][0] for s_i in np_paths_states[i]])
            vel0_i = np.array([s_i[0][1] for s_i in np_paths_states[i]])
            assert len(np.unique(pos0_i)) == 1
            assert len(np.unique(vel0_i)) == 1

            # within a top-level sample, assert all controls are different
            acc0_i = np.array([c_i[0][0] for c_i in np_paths_ctrls[i]])
            assert len(np.unique(np.around(acc0_i,8))) == n_sub_samples

        # ~~~ ASSERT ~~~

        ###
        # check that no top-level states are repeated
        ###
        top_level_init_state_samples = [s[0][0] for s in np_paths_states]
        assert len(top_level_init_state_samples) == n_top_samples_per_iter
        top_level_init_pos_samples = np.array([s[0] for s in top_level_init_state_samples])
        top_level_init_vel_samples = np.array([s[1] for s in top_level_init_state_samples])
        nonunique_top_level_init_pos = get_non_unique_data(top_level_init_pos_samples)
        nonunique_top_level_init_vel = get_non_unique_data(top_level_init_vel_samples)

        if len(nonunique_top_level_init_pos) != 0:
            prnt_str = ""
            for nud in nonunique_top_level_init_pos:
                prnt_str += "Iter {}: Value {} found at non-unique indices {}\n".format(iter, nud[0], nud[1])
            assert False, prnt_str

        if len(nonunique_top_level_init_vel) != 0:
            prnt_str = ""
            for nud in nonunique_top_level_init_vel:
                prnt_str += "Iter {}: Value {} found at non-unique indices {}\n".format(iter, nud[0], nud[1])
            assert False, prnt_str

        ###
        # check that no top-level controls are repeated
        ###

        n_top_level_ctrl_samples = n_top_samples_per_iter*n_sub_samples
        top_level_ctrl_samples = np.concatenate([ccc[0] for ccc in [cc for c in np_paths_ctrls for cc in c]])
        assert top_level_ctrl_samples.size == n_top_level_ctrl_samples
        
        # find any non-unique top-level control samples
        nonunique_top_ctrl_data = get_non_unique_data(top_level_ctrl_samples)
        if len(nonunique_top_ctrl_data) != 0:
            prnt_str = ""
            for nud in nonunique_top_ctrl_data:
                prnt_str += "Iter {}: Value {} found at non-unique indices {}\n".format(iter, nud[0], nud[1])
            assert False, prnt_str

        ###
        # check that no repeated state samples
        ###

        # all states sampled/explored not including top-level sampled states
        n_subsequent_states = n_top_samples_per_iter*n_sub_samples*(n_steps_per_path-1)
        all_subsequent_states = np.concatenate([sss[1:] for sss in [ss for s in np_paths_states for ss in s]])
        assert len(all_subsequent_states) == n_subsequent_states

        # separate the positon and velocity values from all subsequent states
        all_sub_pos = np.array([s[0] for s in all_subsequent_states])
        all_sub_vel = np.array([s[1] for s in all_subsequent_states])

        # find any non-unique position or velocity values
        nonunique_sub_pos_data = get_non_unique_data(np.around(all_sub_pos,8))
        nonunique_sub_vel_data = get_non_unique_data(np.around(all_sub_vel,8))
        if len(nonunique_sub_pos_data) != 0:
            prnt_str = ""
            for nud in nonunique_sub_pos_data:
                prnt_str += "Iter {}: Value {} found at non-unique indices {}\n".format(iter, nud[0], nud[1])
            assert False, prnt_str
        if len(nonunique_sub_vel_data) != 0:
            prnt_str = ""
            for nud in nonunique_sub_vel_data:
                prnt_str += "Iter {}: Value {} found at non-unique indices {}\n".format(iter, nud[0], nud[1])
            assert False, prnt_str

    

def test_sample_risk_metrics_min_risk_controls_0():
    """Check that min-risk control values are not same between samples"""

    # ~~~ ARRANGE ~~~
    # create double integrator object
    pos_bounds = [-32, 32]
    vel_bounds = [-8, 8]
    acc_bounds = [-4, 4] # deterministic system with no accel
    obst_bounds = [1.0, 1.00001]    # very slim obstacle
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # create configuration object as simple namespace
    cfg_obj = SimpleNamespace()
    cfg_obj.n_cores = 4
    cfg_obj.maxtasks = 1
    cfg_obj.n_samples = 4
    cfg_obj.duration = 2.0
    cfg_obj.n_branches = 2
    cfg_obj.tree_depth = 1
    cfg_obj.n_steps = 8
    cfg_obj.policy = "uniform_random"

    # update pickler to allow parallelization of ompl objects
    update_pickler_RealVectorStateSpace2()

    # ~~~ ACT ~~~
    # sample risk metrics
    risk_data = list(sample_risk_metrics(sysset=ds, cfg_obj=cfg_obj, multiproc=True))

    # unpack data
    state_samples, risk_metrics, observations, min_risk_ctrls, min_risk_ctrl_durs = zip(*risk_data)
    min_risk_ctrls = [mrc[0] for mrc in min_risk_ctrls]

    # ~~~ ASSERT ~~~
    # check that no repeated min-risk controls since
    # ctrls are supposed to be uniformly random from the acc_bounds
    # therefore VERY UNLIKELY any two min-risk controls would happen to 
    # be the same
    min_risk_ctrls_duplicates = np.diff(np.sort(min_risk_ctrls))==0
    assert not np.any(min_risk_ctrls_duplicates)

    # ~~~ ACT ~~~
    # run a second time because that matters for some reason!!!
    # sample risk metrics
    risk_data = list(sample_risk_metrics(sysset=ds, cfg_obj=cfg_obj, multiproc=True))

    # unpack data
    state_samples, risk_metrics, observations, min_risk_ctrls, min_risk_ctrl_durs = zip(*risk_data)
    min_risk_ctrls = [mrc[0] for mrc in min_risk_ctrls]

    # ~~~ ASSERT ~~~
    # check that no repeated min-risk controls since
    # ctrls are supposed to be uniformly random from the acc_bounds
    # therefore VERY UNLIKELY any two min-risk controls would happen to 
    # be the same
    min_risk_ctrls_duplicates = np.diff(np.sort(min_risk_ctrls))==0
    assert not np.any(min_risk_ctrls_duplicates)

def sample_DoubleIntegrator1D_control_numpy(dummy, sysset):
    return sysset.sample_control_numpy()

def np_seeder():
    """function called at each worker initialization to ensure numpy gets a new seed
    for each process
    """
    np.random.seed()

def test_multiprocess_sample_control_numpy_0():
    """Check that multiprocess control sampling doesn't produce all the same value
    """

    # ~~~ ARRANGE ~~~
    # create double integrator object
    pos_bounds = [-32, 32]
    vel_bounds = [-8, 8]
    acc_bounds = [-4, 4] # deterministic system with no accel
    obst_bounds = [1.0, 1.00001]    # very slim obstacle
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # create multiprocess task pool
    n_cores = 4
    n_samples = 8
    for i in range(4):
        pool = multiprocess.Pool(n_cores, initializer=np_seeder, maxtasksperchild=1)

        # create partial function
        sample_DoubleIntegrator1D_control_partial = partial(
            sample_DoubleIntegrator1D_control_numpy,
            sysset = ds
        )

        # ~~~ ACT ~~~
        # call control sampler in interative map
        # multiprocess implementation of parallel risk metric estimation
        # use iterative map for process tracking
        ctrl_sampler_iter = pool.imap(sample_DoubleIntegrator1D_control_partial, np.arange(n_samples))

        # track multiprocess progress
        ctrl_samples = []
        for i in range(n_samples):
            ctrl_samples.append(ctrl_sampler_iter.next())

        pool.close()
        pool.join()

        # repackage ctrl samples
        ctrl_samples = [cs[0] for cs in ctrl_samples]

        # ~~~ ASSERT ~~~
        # check that all controls are different (it would be VERY unlikely for any to be 
        # the same for a uniform random sampling)
        ctrl_duplicates = np.diff(np.sort(ctrl_samples))==0
        assert not np.any(ctrl_duplicates)

def sample_DoubleIntegrator1D_control_ompl(dummy, sysset):
    csampler = sysset.space_info.allocControlSampler()
    c = sysset.space_info.allocControl()
    csampler.sample(c)
    return sysset.control_ompl_to_numpy(c)

def notest_multiprocess_sample_control_ompl_0():
    """Check that multiprocess control sampling doesn't produce all the same value

    Note: This is an example of a test that fails by kinda-reproducing the repeat-control-sampling
    bug (that has not been directly resolved, just worked around). Stranely, if you move this to
    the top of the test list, it won't fail if you just run `pytest tests/test_sampler.py`, even
    further highlighting the strangeness of this multiprocess bug. 
    Keeping this test  here for posterity
    but not including it in the pytest process by naming it "notest"
    
    """

    # ~~~ ARRANGE ~~~
    # create double integrator object
    pos_bounds = [-32, 32]
    vel_bounds = [-8, 8]
    acc_bounds = [-4, 4] # deterministic system with no accel
    obst_bounds = [1.0, 1.00001]    # very slim obstacle
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # create multiprocess task pool
    n_cores = 4
    n_samples = 8
    for i in range(4):
        pool = multiprocess.Pool(n_cores, initializer=np_seeder, maxtasksperchild=1)

        # create partial function
        sample_DoubleIntegrator1D_control_partial = partial(
            sample_DoubleIntegrator1D_control_ompl,
            sysset = ds
        )

        # ~~~ ACT ~~~
        # call control sampler in interative map
        # multiprocess implementation of parallel risk metric estimation
        # use iterative map for process tracking
        ctrl_sampler_iter = pool.imap(sample_DoubleIntegrator1D_control_partial, np.arange(n_samples))

        # track multiprocess progress
        ctrl_samples = []
        for i in range(n_samples):
            ctrl_samples.append(ctrl_sampler_iter.next())

        pool.close()
        pool.join()

        # repackage ctrl samples
        ctrl_samples = [cs[0] for cs in ctrl_samples]

        # ~~~ ASSERT ~~~
        # check that all controls are different (it would be VERY unlikely for any to be 
        # the same for a uniform random sampling)
        ctrl_duplicates = np.diff(np.sort(ctrl_samples))==0
        assert not np.any(ctrl_duplicates)

if __name__ == "__main__":
    # test_multiprocess_sample_control_0()
    # test_sample_risk_metrics_min_risk_controls_0()
    test_multiprocess_sampleReachableSet_0()
