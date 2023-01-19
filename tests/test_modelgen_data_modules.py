import torch
import numpy as np

from types import SimpleNamespace

import pyrmm.modelgen.data_modules as DM
from pyrmm.modelgen.double_integrator import local_states_datagen
from pyrmm.setups.double_integrator import DoubleIntegrator1DSetup

def test_DoubleIntegrator1DDataModule_local_states_datagen_0():
    """check state pairing and data generation executes as expected for very small datasets"""
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

    # allocate ompl states
    n_data = 3
    oss = [ds.space_info.allocState() for i in range(n_data)]
    nss = np.array([
        [0,0],
        [0.5, 0.5],
        [1.0, 1.0]
    ])
    for i in range(n_data):
        ds.state_numpy_to_ompl(nss[i],oss[i])

    # create dictionary of simple sample-risk-observation data
    # in separated raw data dictionary 
    srd = dict()
    srd['env1'] = [
        (oss[0], 0.1,  np.array([1e3, 1.0, 0.0]), None, None),
        (oss[1], 0.5,  np.array([1e3, 0.5, 0.5]), None, None),
        (oss[2], 1.0,  np.array([1e3, 0.0, 1.0]), None, None)
    ]

    # ~~~ ACT ~~~
    # perform local state datagen process
    lsd = local_states_datagen(srd)

    # ~~~ ASSERT ~~~
    assert lsd.n_data == 7
    assert np.allclose(lsd.state_samples[0], [0, 0])
    assert np.allclose(lsd.state_samples[1], [-0.5, -0.5])
    assert np.allclose(lsd.state_samples[2], [0.5, 0.5])
    assert np.allclose(lsd.state_samples[3], [0, 0])
    assert np.allclose(lsd.state_samples[4], [-0.5, -0.5])
    assert np.allclose(lsd.state_samples[5], [0.5, 0.5])
    assert np.allclose(lsd.state_samples[6], [0, 0])
    assert np.isclose(lsd.risk_metrics[0], srd['env1'][0][1])
    assert np.isclose(lsd.risk_metrics[1], srd['env1'][0][1])
    assert np.isclose(lsd.risk_metrics[2], srd['env1'][1][1])
    assert np.isclose(lsd.risk_metrics[3], srd['env1'][1][1])
    assert np.isclose(lsd.risk_metrics[4], srd['env1'][1][1])
    assert np.isclose(lsd.risk_metrics[5], srd['env1'][2][1])
    assert np.isclose(lsd.risk_metrics[6], srd['env1'][2][1])
    assert np.allclose(lsd.observations[0], srd['env1'][0][2])
    assert np.allclose(lsd.observations[1], srd['env1'][0][2])
    assert np.allclose(lsd.observations[2], srd['env1'][1][2])
    assert np.allclose(lsd.observations[3], srd['env1'][1][2])
    assert np.allclose(lsd.observations[4], srd['env1'][1][2])
    assert np.allclose(lsd.observations[5], srd['env1'][2][2])
    assert np.allclose(lsd.observations[6], srd['env1'][2][2])


def notest_LocalStateFeatureObservationRiskDataset_getitem_0():
    """test indexing, feature, and coordinate maps work on simple dataset"""

    # ~~~ ARRANGE ~~~
    # define simple state-risk-observation data
    # 1-D state, observation=state, risk=sigmoid(state)
    n_data = 64
    state_samples = np.random.rand(n_data)  # intended range of 0-1 so that MinMaxScaler does not modify (minimally modifies) inputs 
    observations = state_samples
    risk_metrics = torch.sigmoid(torch.tensor(state_samples)).numpy()
    sro_data = DM.BaseRiskMetricTrainingData(
        state_samples=state_samples.reshape(-1,1),
        risk_metrics=risk_metrics.reshape(-1,1),
        observations=observations.reshape(-1,1))

    # define the local coordinate map
    local_coord_map = lambda abs_state, ref_state: abs_state-ref_state

    # define feature vector map
    state_feature_map = lambda x: x

    # ~~~ ACT ~~~
    # create LocalStateFeatureObservationRiskDataset object
    lsfor_dataset = DM.LocalStateFeatureObservationRiskDataset(
        sro_data=sro_data,
        state_feature_map=state_feature_map,
        local_coord_map=local_coord_map)

    # pull data from dataset with __getitem___

    # ~~~ ASSERT ~~~
    # check data from dataset matches expectations
    for i in range(n_data**2):
        abs_idx, ref_idx = np.unravel_index(i, (n_data, n_data))
        s_abs = state_samples[abs_idx]
        s_ref = state_samples[ref_idx]
        ls_i, f_i, o_i, r_i = lsfor_dataset[i]
        assert np.isclose(ls_i, s_abs-s_ref)
        assert np.isclose(f_i, ls_i)
        assert np.isclose(o_i, observations[abs_idx])
        assert np.isclose(r_i, risk_metrics[abs_idx])

def notest_LocalStateFeatureObservationRiskDataset_getitem_di1d_0():
    """test indexing, feature, and coordinate maps work on 1d double-integrator dataset"""
    
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

    # define 1D double integrator state-risk-observation data
    n_data = 64
    ompl_state_samples = n_data*[None]
    np_state_samples = n_data*[None]
    observations = n_data*[None]
    risk_metrics = n_data*[None]
    sampler = ds.space_info.allocStateSampler()
    # create configuration for risk metric estimator
    rm_cfg = SimpleNamespace()
    rm_cfg.duration = 2.0
    rm_cfg.n_branches = 2
    rm_cfg.tree_depth = 1
    rm_cfg.n_steps = 4
    for i in range(n_data):

        # assign state
        ompl_state_samples[i] = ds.space_info.allocState()

        # sample only valid states
        while True:
            sampler.sampleUniform(ompl_state_samples[i])
            if ds.space_info.isValid(ompl_state_samples[i]):
                break

        # get observation from state
        observations[i] = ds.observeState(ompl_state_samples[i])

        # compute risk metric
        risk_metrics[i], _, _ = ds.estimateRiskMetric(
            ompl_state_samples[i],
            None,
            rm_cfg.duration,
            rm_cfg.n_branches,
            rm_cfg.tree_depth,
            rm_cfg.n_steps
            )
        
        # convert state to numpy for ease of processing
        np_state_samples[i] = DoubleIntegrator1DSetup.state_ompl_to_numpy(ompl_state_samples[i])

    # package up state-risk-observation data
    sro_data = DM.BaseRiskMetricTrainingData(
        state_samples=np_state_samples,
        risk_metrics=risk_metrics,
        observations=observations)

    # define the local coordinate map
    local_coord_map = lambda abs_state, ref_state: abs_state-ref_state

    # define feature vector map
    state_feature_map = lambda xi : np.array([xi[0], xi[1], xi[0]**2, xi[1]**2, np.sqrt(2)*xi[0]*xi[1]])

    # ~~~ ACT ~~~
    # create LocalStateFeatureObservationRiskDataset object
    lsfor_dataset = DM.LocalStateFeatureObservationRiskDataset(
        sro_data=sro_data,
        state_feature_map=state_feature_map,
        local_coord_map=local_coord_map)

    # ~~~ ASSERT ~~~
    # check data from dataset matches expectations
    # check data from dataset matches expectations
    for i in range(n_data**2):
        abs_idx, ref_idx = np.unravel_index(i, (n_data, n_data))
        s_abs = np_state_samples[abs_idx]
        s_ref = np_state_samples[ref_idx]
        ls_i, f_i, o_i, r_i = lsfor_dataset[i]
        ls_i_exp = s_abs-s_ref
        assert np.allclose(ls_i,  ls_i_exp)
        assert np.allclose(f_i, state_feature_map(ls_i_exp))
        assert np.allclose(o_i, observations[abs_idx])
        assert np.isclose(r_i, risk_metrics[abs_idx])

if __name__ == "__main__":
    test_DoubleIntegrator1DDataModule_local_states_datagen_0()

