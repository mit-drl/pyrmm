import multiprocess
import numpy as np
from functools import partial
from types import SimpleNamespace

from pyrmm.datagen.sampler import sample_risk_metrics
from pyrmm.setups.double_integrator import DoubleIntegrator1DSetup, \
    update_pickler_RealVectorStateSpace2

def sample_DoubleIntegrator1D_control(dummy, sysset):
    csampler = sysset.space_info.allocControlSampler()
    c = sysset.space_info.allocControl()
    csampler.sample(c)
    return sysset.control_ompl_to_numpy(c)

def test_multiprocess_sample_control_0():
    """Check that multiprocess control sampling doesn't produce all the same value"""

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
        pool = multiprocess.Pool(n_cores, maxtasksperchild=1)

        # create partial function
        sample_DoubleIntegrator1D_control_partial = partial(
            sample_DoubleIntegrator1D_control,
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

    # create multiprocess task pool
    n_cores = 4
    n_top_samples_per_iter = 4
    n_sub_samples = 8
    n_iters = 4
    distance = 2.0
    for i in range(n_iters):
        pool = multiprocess.Pool(n_cores, maxtasksperchild=1)

        # create set of top-level state samples for this iteration
        sampler = ds.space_info.allocStateSampler()
        states = n_top_samples_per_iter * [None] 
        for i in range(n_top_samples_per_iter):

            # assign state
            states[i] = ds.space_info.allocState()

            # sample only valid states
            while True:
                sampler.sampleUniform(states[i])
                if ds.space_info.isValid(states[i]):
                    break

        # create partial function
        sampleReachableSet_partial = partial(
            ds.sampleReachableSet,
            distance = distance, 
            n_samples = n_sub_samples
        )

        # ~~~ ACT ~~~
        # call reachable set sampler
        reachset_samples_iter = pool.imap(sampleReachableSet_partial, states)

        # track multiprocess progress
        reachset_samples = []
        for i in range(n_sub_samples):
            reachset_samples.append(reachset_samples_iter.next())

        pool.close()
        pool.join()

    

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

if __name__ == "__main__":
    # test_multiprocess_sample_control_0()
    test_sample_risk_metrics_min_risk_controls_0()