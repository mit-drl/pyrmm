import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, make_config, instantiate

import hashlib
import git
import yaml
from collections import OrderedDict
from pathlib import Path

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

    # configs = OrderedDict()
    
    # get source file and repo paths
    src_fpath = str(Path(__file__).resolve())
    repo = git.Repo(search_parent_directories=True)
    repo_path = repo.git_dir[:-len('.git')]
    src_fname = str(Path(__file__).stem)
    # configs['src_file']= src_fpath[len(repo_dir):]

    # get git repo hash for file naming
    repo_hash = repo.head.object.hexsha
    # configs['git_hash'] = repo_hash

    # hash file contents
    with open (src_fpath, 'r') as thisfile:
        file_str=thisfile.readlines()
    file_str = ''.join([i for i in file_str])
    file_hash = hashlib.sha1()
    file_hash.update(file_str.encode())
    # configs['file_hash'] = file_hash.hexdigest()

    # create filename
    save_fname = (
        src_fname + '_' +
        repo_hash[:_HASH_LEN] + '_' + 
        file_hash.hexdigest()[:_HASH_LEN])

    return save_fname, repo_path

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

# DubinsPPMSetup Config
_DEFAULT_SPEED = 10.0
_DEFAULT_MIN_TURN_RADIUS = 50.0
_DEFAULT_PPM_FILE = "tests/border_640x400.ppm"
_, repo_path = get_file_info()
DubinsPPMSetupConfig = builds(DubinsPPMSetup, populate_full_signature=True,
    ppm_file = str(Path(repo_path).joinpath(_DEFAULT_PPM_FILE)),
    speed=_DEFAULT_SPEED,
    min_turn_radius=_DEFAULT_MIN_TURN_RADIUS
)

# Top-level configuration and store for command line interface
Config = make_config(
    dubins_ppm_setup=DubinsPPMSetupConfig
)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################
@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config):
    '''Instantiate Dubins setup and generate risk metric data'''
    obj = instantiate(cfg)

if __name__ == "__main__":
    task_function()
