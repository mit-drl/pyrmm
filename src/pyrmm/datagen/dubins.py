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

from ompl import base as ob

from pyrmm.setups.dubins import DubinsPPMSetup
from pyrmm.utils.utils import format_save_filename, get_repo_path

_HASH_LEN = 5
_CONFIG_NAME = "dubins_datamaker_app"
_MONITOR_RATE = 10

##############################################
################# UTILITIES ##################
##############################################

_SAVE_FNAME = format_save_filename(Path(__file__), _HASH_LEN)
_REPO_PATH = get_repo_path()

_DUMMY_SE2SPACE = ob.SE2StateSpace()

def _pickle_SE2StateInternal(state):
    x = state.getX()
    y = state.getY()
    yaw = state.getYaw()
    return _unpickle_SE2StateInternal, (x, y, yaw)

def _unpickle_SE2StateInternal(x, y, yaw):
    state = _DUMMY_SE2SPACE.allocState()
    state.setX(x)
    state.setY(y)
    state.setYaw(yaw)
    return state

def update_pickler():
    '''updates pickler to enable pickling and unpickling of ompl objects'''
    copyreg.pickle(_DUMMY_SE2SPACE.SE2StateInternal, _pickle_SE2StateInternal, _unpickle_SE2StateInternal)


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
Config = make_config(
    setup=DubinsPPMSetupConfig,
    n_samples=zf(int, _DEFAULT_N_SAMPLES),
    duration=zf(float, _DEFAULT_DURATION),
    n_branches=zf(int, _DEFAULT_N_BRANCHES),
    tree_depth=zf(int,_DEFAULT_TREE_DEPTH),
    n_steps=zf(int,_DEFAULT_N_STEPS),
    policy=zf(str,_DEFAULT_POLICY)
)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config):
    '''Instantiate Dubins setup and generate risk metric data'''

    obj = instantiate(cfg)
    num_cores = multiprocess.cpu_count()

    # sample states to evaluate risk metrics
    sampler = obj.setup.space_info.allocValidStateSampler()
    ssamples = obj.n_samples * [None] 
    for i in range(obj.n_samples):

        # assign state
        ssamples[i] = obj.setup.space_info.allocState()
        sampler.sample(ssamples[i])

    # multiprocess implementation of parallel risk metric estimation
    update_pickler()
    partial_estimateRiskMetric = partial(
        obj.setup.estimateRiskMetric, 
        trajectory=None,
        distance=obj.duration,
        branch_fact=obj.n_branches,
        depth=obj.tree_depth,
        n_steps=obj.n_steps,
        policy=obj.policy
    )

    # use iterative map for process tracking
    t_start = time.time()
    rmetrics = multiprocess.Pool(num_cores).imap(partial_estimateRiskMetric, ssamples)

    # track multiprocess progress
    for i,_ in enumerate(ssamples):
        rmetrics.next()
        if i%_MONITOR_RATE ==  0:
            print("{} of {} completed after {:.2f} seconds".format(i, len(ssamples), time.time()-t_start))

    print("total time: {:.2f}".format(time.time()-t_start))

    # save data for pytorch training
    data = [i for i in zip(ssamples, rmetrics)]
    torch.save(data, open(_SAVE_FNAME+".pt", "wb"))

if __name__ == "__main__":
    task_function()
