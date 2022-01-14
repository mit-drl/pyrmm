'''
Utility functions and classes for risk metric maps

Examples: collision checkers, tree and node definitions
'''

import functools
import hashlib
import git
import copyreg
import ompl.base
from typing import Type, List
from pathlib import PosixPath

# Standardized naming variables
SYSTEM_SETUP = 'system_setup'
N_SAMPLES = 'n_samples'
DURATION = 'duration'
N_BRANCHES = 'n_branches'
TREE_DEPTH = 'tree_depth'
N_STEPS = 'n_steps'
POLICY = 'policy'
N_CORES = 'n_cores'


def get_repo_path():
    '''get full path to repo head'''
    repo = git.Repo(search_parent_directories=True)
    repo_path = repo.git_dir[:-len('.git')]
    return repo_path

def format_save_filename(src_file: Type[PosixPath], hash_len: int):
    '''create the save filename using timestamps and hashes

    Args:
        fpath : PosixPath
            pathlib Path to source file to format for output data
        hash_len : int
            length of hash to use in filename
    
    Returns:
        save_fname : str
            directory name of source file (i.e. datagen to distinguish from trained models)
            string desriptor of the datatype (name of source file)
            the git hash of the repo state
            the sha1 hash of this files contents (may change between repo commits)
            SRCFILE_REPOHASH_FILEHASH
    '''

    
    # get source file and repo paths

    src_fpath = str(src_file.resolve())
    src_dname = str(src_file.parts[-2])
    src_fname = str(src_file.stem)

    # get git repo hash for file naming
    repo = git.Repo(search_parent_directories=True)
    repo_hash = repo.head.object.hexsha

    # hash file contents
    with open (src_fpath, 'r') as thisfile:
        file_str=thisfile.readlines()
    file_str = ''.join([i for i in file_str])
    file_hash = hashlib.sha1()
    file_hash.update(file_str.encode())

    # create filename
    save_fname = (
        src_dname + '_' +
        src_fname + '_' +
        repo_hash[:hash_len] + '_' + 
        file_hash.hexdigest()[:hash_len])

    return save_fname

_DUMMY_SE2SPACE = ompl.base.SE2StateSpace()

def _pickle_SE2StateInternal(state):
    '''pickle OMPL SE2StateInternal object'''
    x = state.getX()
    y = state.getY()
    yaw = state.getYaw()
    return _unpickle_SE2StateInternal, (x, y, yaw)

def _unpickle_SE2StateInternal(x, y, yaw):
    '''unpickle OMPL SE2StateInternal object'''
    state = _DUMMY_SE2SPACE.allocState()
    state.setX(x)
    state.setY(y)
    state.setYaw(yaw)
    return state

def update_pickler_se2stateinternal():
    '''updates pickler to enable pickling and unpickling of ompl objects'''
    copyreg.pickle(_DUMMY_SE2SPACE.SE2StateInternal, _pickle_SE2StateInternal, _unpickle_SE2StateInternal)

def is_pixel_free_space(p):
    '''check if pixel is in free space of obstacles'''
    tr = p.red > 127
    tg = p.green > 127
    tb = p.green > 127
    return tr and tg and tb

def partialclass(cls, *args, **kwds):
    '''Ref: https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor'''

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls


class Node2D:
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = [x]
        self.path_y = [y]
        self.parent = None

def check_collision_2D_circular_obstacles(node: Node2D, obstacleList: List):
    ''' check if node, and path to node, is in collision with any circular obstacles

    Args:
        node: Node
            state node to check for collisions, including path to the obstacle
        obstacleList: List
            list of circular obstacles, each formated as (x-pos, y-pos, radius)
        
    Returns:
        is_collision: bool
            True if node is in collision with any obstacle, otherwise false
    '''

    if node is None:
        return True

    for (ox, oy, size) in obstacleList:
        dx_list = [ox - x for x in node.path_x]
        dy_list = [oy - y for y in node.path_y]
        d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

        if min(d_list) <= size**2:
            return True  # collision

    return False  # safe