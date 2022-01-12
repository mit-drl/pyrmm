import hydra
import torch
import hashlib
import git
import copyreg
import multiprocess
from functools import partial
from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, instantiate, make_custom_builds_fn
from hydra_zen import ZenField as zf

from ompl import base as ob

from pyrmm.setups.dubins import DubinsPPMSetup

_HASH_LEN = 5
_CONFIG_NAME = "dubins_datamaker_app"

# Setup argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument('--speed', type=float, default=10.0, help='[m/s] speed of dubins vehicle')
# parser.add_argument('--min-turn-radius', type=float, default=50.0, help='[m] minimum turning radius of dubins vehicle')
# parser.add_argument('--obstacle-file', type=str, default='tests/border_640x400.ppm', help='path to ppm file representing obstacles')
# parser.add_argument('--policy', type=str, default='uniform_random', help='description of policy used for control propagation')
# parser.add_argument('--duration', type=float, default=2.0, help='[s] duration of policy propagation at each depth')
# parser.add_argument('--tree-depth', type=int, default=4, help='number of depth layers in propagation tree')
# parser.add_argument('--tree-branching', type=int, default=16, help='number of branches per depth in control propagation tree')
# parser.add_argument('--n-steps', type=int, default=2, help='number of intermediate steps in each control segment for collision checking')
# parser.add_argument('--n-samples', type=int, default=1024, help='number of top-level samples to draw for data generation')

def get_file_info():
    '''create the save filename using timestamps and hashes
    
    Returns:
        save_fname : str
            string desriptor of the datatype (name of source file)
            the git hash of the repo state
            the sha1 hash of this files contents (may change between repo commits)
            SRCFILE_REPOHASH_FILEHASH
        repo_path : str
            full path to repo head
    '''

    
    # get source file and repo paths
    src_fpath = str(Path(__file__).resolve())
    repo = git.Repo(search_parent_directories=True)
    repo_path = repo.git_dir[:-len('.git')]
    src_fname = str(Path(__file__).stem)

    # get git repo hash for file naming
    repo_hash = repo.head.object.hexsha

    # hash file contents
    with open (src_fpath, 'r') as thisfile:
        file_str=thisfile.readlines()
    file_str = ''.join([i for i in file_str])
    file_hash = hashlib.sha1()
    file_hash.update(file_str.encode())

    # create filename
    save_fname = (
        src_fname + '_' +
        repo_hash[:_HASH_LEN] + '_' + 
        file_hash.hexdigest()[:_HASH_LEN])

    return save_fname, repo_path

_SAVE_FNAME, _REPO_PATH = get_file_info()

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

# def state_sampler_inputs_config(n_samples:int=_DEFAULT_N_SAMPLES):
#     return {'n_samples': n_samples}
# StateSamplerInputsConfig = pbuilds(state_sampler_inputs_config)

# def risk_estimator_inputs(
#     duration:float=_DEFAULT_DURATION, 
#     n_branches:int=_DEFAULT_N_BRANCHES, 
#     tree_depth:int=_DEFAULT_TREE_DEPTH, 
#     n_steps:int=_DEFAULT_N_STEPS, 
#     policy:str=_DEFAULT_POLICY):
#     return {'duration': duration, 'n_branches': n_branches, 'tree_depth': tree_depth, 'n_steps': n_steps, 'policy': policy}
# RiskEstimatorConfig = pbuilds(risk_estimator_inputs)

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
    partial_estimateRiskMetric = partial(
        obj.setup.estimateRiskMetric, 
        trajectory=None,
        distance=obj.duration,
        branch_fact=obj.n_branches,
        depth=obj.tree_depth,
        n_steps=obj.n_steps,
        policy=obj.policy
    )
    with multiprocess.Pool(num_cores) as pool:
        rmetrics = pool.map(partial_estimateRiskMetric, ssamples)

    # save data for pytorch training
    data = [i for i in zip(ssamples, rmetrics)]
    torch.save(data, open(_SAVE_FNAME+".pt", "wb"))

if __name__ == "__main__":
    task_function()
