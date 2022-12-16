# tools to assist sampling of states and computing risk metrics in
# data generation process

import time
import multiprocess as mp
from functools import partial

import pyrmm.utils.utils as U
from pyrmm.setups import SystemSetup

_MONITOR_RATE = 100

def sample_risk_metrics(sysset:SystemSetup, cfg_obj, multiproc:bool=True, prfx:str=''):
    '''sample system states, get observations, and estimate risk metrics
    
    Args:
        sysset : SystemSetup
            the vehicle-config space setup object to be used
            for sampling and risk metric computation
        cfg_obj :
            instantiate configuration object
        multiproc : bool
            If true, use multi-process implemtnation
        prfx : string
            string to add to prefix print statements

    Returns:
        risk_data : List
            Each element in list is a tuple (omplState, risk_metric, observation)
    '''

    if multiproc:
        pool = mp.Pool(getattr(cfg_obj, U.N_CORES), maxtasksperchild=cfg_obj.maxtasks)

    # sample states to evaluate risk metrics
    sampler = sysset.space_info.allocStateSampler()
    states = getattr(cfg_obj, U.N_SAMPLES) * [None] 
    observations = getattr(cfg_obj, U.N_SAMPLES) * [None] 
    for i in range(getattr(cfg_obj, U.N_SAMPLES)):

        # assign state
        states[i] = sysset.space_info.allocState()

        # sample only valid states
        while True:
            sampler.sampleUniform(states[i])
            if sysset.space_info.isValid(states[i]):
                break

        # get observation from state
        observations[i] = sysset.observeState(states[i])

        if i%_MONITOR_RATE ==  0:
            print("{}State sampling and ray casting: completed {} of {}".format(prfx, i, len(states)))

    if hasattr(sysset, 'base_risk_estimator'):
        base_estimate_fn = partial(sysset.base_risk_estimator)
    else:
        base_estimate_fn = None

    partial_estimateRiskMetric = partial(
            sysset.estimateRiskMetric, 
            trajectory=None,
            distance=getattr(cfg_obj, U.DURATION),
            branch_fact=getattr(cfg_obj, U.N_BRANCHES),
            depth=getattr(cfg_obj, U.TREE_DEPTH),
            n_steps=getattr(cfg_obj, U.N_STEPS),
            policy=getattr(cfg_obj, U.POLICY),
            base_estimate_fn=base_estimate_fn
        )
    t_start = time.time()

    if multiproc: 
        # multiprocess implementation of parallel risk metric estimation
        # use iterative map for process tracking
        rmetrics_iter = pool.imap(partial_estimateRiskMetric, states)

        # track multiprocess progress
        risk_n_ctrl_tuples = []
        for i,_ in enumerate(states):
            risk_n_ctrl_tuples.append(rmetrics_iter.next())
            if i%_MONITOR_RATE ==  0:
                print("{}Risk metric evaluation: completed {} of {} after {:.2f}".format(prfx, i, len(states), time.time()-t_start))

        pool.close()
        pool.join()

    else:
        # single-process implementation of risk metric estimation
        risk_n_ctrl_tuples = []
        for i, state in enumerate(states):
            risk_n_ctrl_tuples.append(partial_estimateRiskMetric(state=state))
            if i%_MONITOR_RATE ==  0:
                print("{}Risk metric evaluation: completed {} of {} after {:.2f}".format(prfx,i, len(states), time.time()-t_start))

    print("{}Total risk estimation elapsed time: {:.2f}".format(prfx,time.time()-t_start))

    # unpack risk metrics and min-risk-ctrl tuples into index-aligned tuples
    risk_metrics, min_risk_ctrls, min_risk_ctrl_durs = list(zip(*risk_n_ctrl_tuples))

    risk_data = zip(states, risk_metrics, observations, min_risk_ctrls, min_risk_ctrl_durs)
    return risk_data

    # # save data for pytorch training
    # data = [i for i in zip(states, risk_metrics, observations)]
    # torch.save(data, open(save_name+".pt", "wb"))
