# tools to assist sampling of states and computing risk metrics in
# data generation process

import torch
import time
import multiprocess
from functools import partial

import pyrmm.utils.utils as U
from pyrmm.setups import SystemSetup

_MONITOR_RATE = 100

def sample_risk_metrics(sysset: SystemSetup, cfg_obj, save_name: str):
    '''sample states of DubinsPPMSetup and compute risk metrics
    
    Args:
        sysset : SystemSetup
            the vehicle-config space setup object to be used
            for sampling and risk metric computation
        cfg_obj :
            instantiate configuration object
        save_name : str
            name of pickle file to save risk metrics
    '''

    pool = multiprocess.Pool(getattr(cfg_obj, U.N_CORES), maxtasksperchild=cfg_obj.maxtasks)

    # sample states to evaluate risk metrics
    sampler = sysset.space_info.allocStateSampler()
    states = getattr(cfg_obj, U.N_SAMPLES) * [None] 
    observations = getattr(cfg_obj, U.N_SAMPLES) * [None] 
    for i in range(getattr(cfg_obj, U.N_SAMPLES)):

        # assign state
        states[i] = sysset.space_info.allocState()
        sampler.sampleUniform(states[i])

        # get observation from state
        observations[i] = sysset.observeState(states[i])

        if i%_MONITOR_RATE ==  0:
            print("State sampling and ray casting: completed {} of {}".format(i, len(states)))

    # multiprocess implementation of parallel risk metric estimation
    partial_estimateRiskMetric = partial(
        sysset.estimateRiskMetric, 
        trajectory=None,
        distance=getattr(cfg_obj, U.DURATION),
        branch_fact=getattr(cfg_obj, U.N_BRANCHES),
        depth=getattr(cfg_obj, U.TREE_DEPTH),
        n_steps=getattr(cfg_obj, U.N_STEPS),
        policy=getattr(cfg_obj, U.POLICY)
    )

    # use iterative map for process tracking
    t_start = time.time()
    rmetrics_iter = pool.imap(partial_estimateRiskMetric, states)

    # track multiprocess progress
    risk_metrics = []
    for i,_ in enumerate(states):
        risk_metrics.append(rmetrics_iter.next())
        if i%_MONITOR_RATE ==  0:
            print("Risk metric evaluation: completed {} of {} after {:.2f}".format(i, len(states), time.time()-t_start))

    pool.close()
    pool.join()

    print("Subprocess elapsed time: {:.2f}".format(time.time()-t_start))

    # save data for pytorch training
    data = [i for i in zip(states, risk_metrics, observations)]
    torch.save(data, open(save_name+".pt", "wb"))