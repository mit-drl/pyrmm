import hydra
import torch
import copyreg
import multiprocess
import time
from functools import partial
from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, instantiate, make_custom_builds_fn
from hydra_zen import ZenField as zf

import pyrmm.utils.utils as U
from pyrmm.setups.dubins import DubinsPPMSetup

_HASH_LEN = 5
_CONFIG_NAME = "dubins_datamaker_app"
_MONITOR_RATE = 10

##############################################
################# UTILITIES ##################
##############################################

_SAVE_FNAME = U.format_save_filename(Path(__file__), _HASH_LEN)
_REPO_PATH = U.get_repo_path()

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(populate_full_signature=True)

# DubinsPPMSetup Config
_DEFAULT_SPEED = 10.0
_DEFAULT_MIN_TURN_RADIUS = 50.0
_DEFAULT_PPM_FILE = "tests/border_640x400.ppm"
DubinsPPMSetupConfig = pbuilds(DubinsPPMSetup,
    ppm_file = str(Path(_REPO_PATH).joinpath(_DEFAULT_PPM_FILE)),
    speed=_DEFAULT_SPEED,
    min_turn_radius=_DEFAULT_MIN_TURN_RADIUS
)

# Default sampler and risk estimator configs
_DEFAULT_N_SAMPLES = 1024
_DEFAULT_DURATION = 2.0
_DEFAULT_N_BRANCHES = 16
_DEFAULT_TREE_DEPTH = 4
_DEFAULT_N_STEPS = 2
_DEFAULT_POLICY = 'uniform_random'

# Top-level configuration and store for command line interface
make_config_input = {
    U.SYSTEM_SETUP: DubinsPPMSetupConfig,
    U.N_SAMPLES: zf(int, _DEFAULT_N_SAMPLES),
    U.DURATION: zf(float, _DEFAULT_DURATION),
    U.N_BRANCHES: zf(int, _DEFAULT_N_BRANCHES),
    U.TREE_DEPTH: zf(int,_DEFAULT_TREE_DEPTH),
    U.N_STEPS: zf(int,_DEFAULT_N_STEPS),
    U.POLICY: zf(str,_DEFAULT_POLICY),
    U.N_CORES: zf(int, multiprocess.cpu_count())
}
Config = make_config(**make_config_input)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config):
    '''Instantiate Dubins setup and generate risk metric data'''

    obj = instantiate(cfg)

    # sample states to evaluate risk metrics
    # sampler = getattr(obj, U.SYSTEM_SETUP).space_info.allocValidStateSampler()
    sampler = getattr(obj, U.SYSTEM_SETUP).space_info.allocStateSampler()
    ssamples = getattr(obj, U.N_SAMPLES) * [None] 
    for i in range(getattr(obj, U.N_SAMPLES)):

        # assign state
        ssamples[i] = getattr(obj, U.SYSTEM_SETUP).space_info.allocState()
        # sampler.sample(ssamples[i])
        sampler.sampleUniform(ssamples[i])

    # multiprocess implementation of parallel risk metric estimation
    U.update_pickler_se2stateinternal()
    # partial_estimateRiskMetric = partial(
    #     getattr(obj, U.SYSTEM_SETUP).estimateRiskMetric, 
    #     trajectory=None,
    #     distance=getattr(obj, U.DURATION),
    #     branch_fact=getattr(obj, U.N_BRANCHES),
    #     depth=getattr(obj, U.TREE_DEPTH),
    #     n_steps=getattr(obj, U.N_STEPS),
    #     policy=getattr(obj, U.POLICY)
    # )
    partial_estimateRiskMetric = partial(
        getattr(obj, U.SYSTEM_SETUP).dummyRiskMetric, 
        trajectory=None,
        distance=getattr(obj, U.DURATION),
        branch_fact=getattr(obj, U.N_BRANCHES),
        depth=getattr(obj, U.TREE_DEPTH),
        n_steps=getattr(obj, U.N_STEPS),
        policy=getattr(obj, U.POLICY)
    )

    # use iterative map for process tracking
    t_start = time.time()
    pool = multiprocess.Pool(getattr(obj, U.N_CORES))
    rmetrics_iter = pool.imap(partial_estimateRiskMetric, ssamples)

    # track multiprocess progress
    rmetrics = []
    for i,_ in enumerate(ssamples):
        rmetrics.append(rmetrics_iter.next())
        if i%_MONITOR_RATE ==  0:
            print("{} of {} completed after {:.2f} seconds".format(i, len(ssamples), time.time()-t_start))

    pool.close()
    pool.join()

    print("total time: {:.2f}".format(time.time()-t_start))

    # save data for pytorch training
    data = [i for i in zip(ssamples, rmetrics)]
    torch.save(data, open(_SAVE_FNAME+".pt", "wb"))

if __name__ == "__main__":
    task_function()
