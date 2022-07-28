# Procedurally generates a set of mazes using https://github.com/razimantv/mazegenerator

import subprocess
import pyvips
import hydra

import pyrmm.utils.utils as U

from hydra_zen import instantiate, make_config
from hydra.core.config_store import ConfigStore


_CONFIG_NAME = "mazegen_app"
# _N_MAZE_TYPES = 5   # see https://github.com/razimantv/mazegenerator#usage

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

_DEFAULT_N_MAZES = 2   # two of each type of maze, by default
_DEFAULT_MAZE_TYPES=None

MazegenConfig = make_config(
    maze_types = _DEFAULT_MAZE_TYPES,
    n_mazes_per_type = _DEFAULT_N_MAZES)

# Store the top level config for command line interface
cs = ConfigStore.instance()
cs.store(_CONFIG_NAME, node=MazegenConfig)

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: MazegenConfig):

    repo_dir = U.get_repo_path()

    # instantiate the mazegen config object
    obj = instantiate(cfg)

    # determine maze types to generate
    if obj.maze_types is None:
        obj.maze_types = list(range(5)) # see https://github.com/razimantv/mazegenerator#usage
    if not all([mt in range(5) for mt in obj.maze_types]):
        raise ValueError('Unexpected maze types in {}. Expect only integers in range [0,4]'.format(obj.maze_types))
    n_maze_types = len(obj.maze_types)

    # iterate through each maze
    i=0
    for maze_type in obj.maze_types:
        for maze_type_cnt in range(obj.n_mazes_per_type):

            # get maze type integer and create maze name
            # maze_type_cnt, maze_type = divmod(i, n_maze_types)
            maze_name = '_'.join(['maze', str(i), 'type', str(maze_type)])
        
            # generate mazes
            mazegen_cmd = repo_dir + '/mazegenerator/src/mazegen'
            mazegen_cmd += ' -o ' + maze_name
            mazegen_cmd += ' -m ' + str(maze_type)
            if maze_type == 2:
                # reduce size for honeycomb mazes because default is too large
                # for memory
                mazegen_cmd += ' -s 10'
            mazegen_proc = subprocess.Popen(mazegen_cmd.split(), stdout=subprocess.PIPE)
            output, error = mazegen_proc.communicate()

            # convert to ppm
            image = pyvips.Image.new_from_file(maze_name+'.svg', dpi=300)
            image.write_to_file(maze_name+'.ppm')

            i+=1

##############################################
############### TASK FUNCTIONS ###############
##############################################

if __name__ == "__main__":
    task_function()
