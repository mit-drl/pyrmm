import numpy as np
from types import SimpleNamespace

from pyrmm.datagen.sampler import sample_risk_metrics
from pyrmm.setups.double_integrator import DoubleIntegrator1DSetup, \
    update_pickler_RealVectorStateSpace2

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
    cfg_obj.n_cores = 8
    cfg_obj.maxtasks = 16
    cfg_obj.n_samples = 32
    cfg_obj.duration = 2.0
    cfg_obj.n_branches = 32
    cfg_obj.tree_depth = 2
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
    test_sample_risk_metrics_min_risk_controls_0()