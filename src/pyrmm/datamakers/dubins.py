import argparse
import hashlib
import git
import yaml
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

HASH_LEN = 5
DESCRIPTOR = 'dubins'
SAVE_DIR = 'data'

# Setup argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--speed', type=float, default=10.0, help='[m/s] speed of dubins vehicle')
parser.add_argument('--min-turn-radius', type=float, default=50.0, help='[m] minimum turning radius of dubins vehicle')
parser.add_argument('--obstacle-file', type=str, default='border_640x400.ppm', help='path to ppm file representing obstacles')
parser.add_argument('--policy', type=str, default='uniform_random', help='description of policy used for control propagation')
parser.add_argument('--duration', type=float, default=2.0, help='[s] duration of policy propagation at each depth')
parser.add_argument('--tree-depth', type=int, default=4, help='number of depth layers in propagation tree')
parser.add_argument('--tree-branching', type=int, default=16, help='number of branches per depth in control propagation tree')
parser.add_argument('--n-steps', type=int, default=2, help='number of intermediate steps in each control segment for collision checking')
parser.add_argument('--n-samples', type=int, default=1024, help='number of top-level samples to draw for data generation')

def get_savefile(inargs):
    '''create the save filename using timestamps and hashes
    
    Returns:
        save_fname : str
            filename is formatted with the datetime when the file was called,
            the git hash of the repo
            string desriptor of the datatype
            the sha1 hash of this files contents (may change between repo commits)
            the sha1 hash of the input args
            YYYYMMDD_HHMMSS_DESC_REPO_FILE_ARGS
        configs : dict
            configurations to be saved into yaml file
    '''

    configs = OrderedDict()
    # configs = dict()

    src_fpath = str(Path(__file__).resolve())
    repo = git.Repo(search_parent_directories=True)
    repo_dir = repo.git_dir[:-len('.git')]
    # configs['src_file']= src_fpath.removeprefix(repo.git_dir.removesuffix('.git'))
    configs['src_file']= src_fpath[len(repo_dir):]

    # get datetime for file naming
    dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    configs['datetime'] = dt_str
    # print(dt_str)

    # get git repo hash for file naming
    repo_hash = repo.head.object.hexsha
    configs['git_hash'] = repo_hash

    # hash file contents
    with open (src_fpath, 'r') as thisfile:
        file_str=thisfile.readlines()
    file_str = ''.join([i for i in file_str])
    file_hash = hashlib.sha1()
    file_hash.update(file_str.encode())
    configs['file_hash'] = file_hash.hexdigest()
    
    # hash input arguments
    args_hash = hashlib.sha1()
    args_hash.update(str(inargs).encode())
    configs['args_hash'] = args_hash.hexdigest()

    # add args to configs
    # configs = {**configs, **vars(args)}
    # configs.update(vars(args))

    # create filename
    save_fname = (
        dt_str + '_' +
        DESCRIPTOR + '_' +
        repo_hash[:HASH_LEN] + '_' + 
        file_hash.hexdigest()[:HASH_LEN] + '_' + 
        args_hash.hexdigest()[:HASH_LEN])

    # save configs as yaml
    config_fname = Path(repo_dir).resolve().joinpath(SAVE_DIR, save_fname+'.yml')
    with open(config_fname, 'w') as config_file:
        # yaml.dump({**configs}, config_file)
        for k, v in configs.items():
            yaml.dump({k:v}, config_file)
        yaml.dump(vars(args), config_file)

    return save_fname


if __name__ == "__main__":

    # get input args
    args = parser.parse_args()

    savefile_prfx = get_savefile(args)
    print(savefile_prfx)
