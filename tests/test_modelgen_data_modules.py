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
    obst_bounds = [2.0, 2.0001]    # very slim obstacle
    ds = DoubleIntegrator1DSetup(
        pos_bounds=pos_bounds, 
        vel_bounds=vel_bounds, 
        acc_bounds=acc_bounds, 
        obst_bounds=obst_bounds)

    # allocate ompl states
    n_data = 6
    oss = [ds.space_info.allocState() for i in range(n_data)]
    nss = np.array([
        [-0.5,0],
        [0.5, 0.5],
        [1.0, 1.0],
        [2.0, 0.0], # boundary case where states 0,1,2 can see 3 and vice versa
        [2.0, 0.0], # boundary case where states 0,1,2 can see 3 but 3 can't see 0,1,2
        [2.5, 0.0]  # other side of obstacle
    ])
    for i in range(n_data):
        ds.state_numpy_to_ompl(nss[i],oss[i])

    # create dictionary of simple sample-risk-observation data
    # in separated raw data dictionary 
    srd = dict()
    srd['env1'] = [
        (oss[0], 0.1,  np.array([1e3, 2.5, 0.0]), None, None),
        (oss[1], 0.25,  np.array([1e3, 1.5, 0.5]), None, None),
        (oss[2], 0.75,  np.array([1e3, 1.0, 1.0]), None, None),
        (oss[3], 1.0,  np.array([1e3, 0.0, 0.0]), None, None), # boundary case where states 0,1,2,4 can see 3 and vice versa
        (oss[4], 1.0,  np.array([0.0, 0.0, 0.0]), None, None), # boundary case where states 0,1,2,3 can see 4 but 4 can't see 0,1,2
        (oss[5], 0.2,  np.array([.4999, 1e3, 0.0]), None, None), # other side of obstacle
    ]
    
    # state 2 is degenerate case with state on the boundary of obstacle
    # where states 0 and 1 can't be "seen" from state 2: i.e. local state 
    # but state 2 can be "seen" from 0 and 1
    # srd['env1'].append(oss[2], 1.0,  np.array([1e3, 0.0, 1.0]), None, None)   

    # ~~~ ACT ~~~
    # perform local state datagen process
    dt = 2.0
    lsd = local_states_datagen(dt, srd)

    # ~~~ ASSERT ~~~
    assert lsd.n_data == 20

    # abs-state 0
    assert np.allclose(lsd.state_samples[0], [0.0, 0.0])
    assert np.allclose(lsd.state_samples[1], [-1.0, -0.5])
    assert np.allclose(lsd.state_samples[2], [-1.5, -1.0])
    assert np.isclose(lsd.risk_metrics[0], srd['env1'][0][1])
    assert np.isclose(lsd.risk_metrics[1], srd['env1'][0][1])
    assert np.isclose(lsd.risk_metrics[2], srd['env1'][0][1])
    assert np.allclose(lsd.observations[0], srd['env1'][0][2])
    assert np.allclose(lsd.observations[1], srd['env1'][0][2])
    assert np.allclose(lsd.observations[2], srd['env1'][0][2])
    
    # abs-state 1
    assert np.allclose(lsd.state_samples[3], [1.0, 0.5])
    assert np.allclose(lsd.state_samples[4], [0.0, 0.0])
    assert np.allclose(lsd.state_samples[5], [-0.5, -0.5])
    assert np.allclose(lsd.state_samples[6], [-1.5, 0.5])
    assert np.allclose(lsd.state_samples[7], [-1.5, 0.5])
    assert np.isclose(lsd.risk_metrics[3], srd['env1'][1][1])
    assert np.isclose(lsd.risk_metrics[4], srd['env1'][1][1])
    assert np.isclose(lsd.risk_metrics[5], srd['env1'][1][1])
    assert np.isclose(lsd.risk_metrics[6], srd['env1'][1][1])
    assert np.isclose(lsd.risk_metrics[7], srd['env1'][1][1])
    assert np.allclose(lsd.observations[3], srd['env1'][1][2])
    assert np.allclose(lsd.observations[4], srd['env1'][1][2])
    assert np.allclose(lsd.observations[5], srd['env1'][1][2])
    assert np.allclose(lsd.observations[6], srd['env1'][1][2])
    assert np.allclose(lsd.observations[7], srd['env1'][1][2])

    # abs-state 2
    assert np.allclose(lsd.state_samples[8], [1.5, 1.0])
    assert np.allclose(lsd.state_samples[9], [0.5, 0.5])
    assert np.allclose(lsd.state_samples[10], [0.0, 0.0])
    assert np.allclose(lsd.state_samples[11], [-1.0, 1.0])
    assert np.allclose(lsd.state_samples[12], [-1.0, 1.0])
    assert np.isclose(lsd.risk_metrics[8], srd['env1'][2][1])
    assert np.isclose(lsd.risk_metrics[9], srd['env1'][2][1])
    assert np.isclose(lsd.risk_metrics[10], srd['env1'][2][1])
    assert np.isclose(lsd.risk_metrics[11], srd['env1'][2][1])
    assert np.isclose(lsd.risk_metrics[12], srd['env1'][2][1])
    assert np.allclose(lsd.observations[8], srd['env1'][2][2])
    assert np.allclose(lsd.observations[9], srd['env1'][2][2])
    assert np.allclose(lsd.observations[10], srd['env1'][2][2])
    assert np.allclose(lsd.observations[11], srd['env1'][2][2])
    assert np.allclose(lsd.observations[12], srd['env1'][2][2])

    # abs-state 3: Boundary Case
    assert np.allclose(lsd.state_samples[13], [1.5, -0.5])
    assert np.allclose(lsd.state_samples[14], [1.0, -1.0])
    assert np.allclose(lsd.state_samples[15], [0.0, 0.0])
    assert np.allclose(lsd.state_samples[16], [0.0, 0.0])
    assert np.isclose(lsd.risk_metrics[13], srd['env1'][3][1])
    assert np.isclose(lsd.risk_metrics[14], srd['env1'][3][1])
    assert np.isclose(lsd.risk_metrics[15], srd['env1'][3][1])
    assert np.isclose(lsd.risk_metrics[16], srd['env1'][3][1])
    assert np.allclose(lsd.observations[13], srd['env1'][3][2])
    assert np.allclose(lsd.observations[14], srd['env1'][3][2])
    assert np.allclose(lsd.observations[15], srd['env1'][3][2])
    assert np.allclose(lsd.observations[16], srd['env1'][3][2])

    # abs-state 4: Degenerate Boundary Case
    assert np.allclose(lsd.state_samples[17], [0.0, 0.0])
    assert np.allclose(lsd.state_samples[18], [0.0, 0.0])
    assert np.isclose(lsd.risk_metrics[17], srd['env1'][4][1])
    assert np.isclose(lsd.risk_metrics[18], srd['env1'][4][1])
    assert np.allclose(lsd.observations[17], srd['env1'][4][2])
    assert np.allclose(lsd.observations[18], srd['env1'][4][2])

    # abs-state 5:
    assert np.allclose(lsd.state_samples[19], [0.0, 0.0])
    assert np.isclose(lsd.risk_metrics[19], srd['env1'][5][1])
    assert np.allclose(lsd.observations[19], srd['env1'][5][2])


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

