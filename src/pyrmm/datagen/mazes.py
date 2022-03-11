# Procedurally generates a set of mazes using https://github.com/razimantv/mazegenerator

import subprocess
import pyvips
import hydra

import pyrmm.utils.utils as U

from hydra_zen import instantiate, make_config
from hydra.core.config_store import ConfigStore


_CONFIG_NAME = "mazegen_app"
_N_MAZE_TYPES = 5   # see https://github.com/razimantv/mazegenerator#usage

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

_DEFAULT_N_MAZES = 10   # two of each type of maze, by default

MazegenConfig = make_config(n_mazes=_DEFAULT_N_MAZES)

# Store the top level config for command line interface
cs = ConfigStore.instance()
cs.store(_CONFIG_NAME, node=MazegenConfig)

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: MazegenConfig):

    repo_dir = U.get_repo_path()

    # instantiate the mazegen config object
    obj = instantiate(cfg)

    # iterate through each maze
    for i in range(obj.n_mazes):

        # get maze type integer and create maze name
        maze_type_cnt, maze_type = divmod(i, _N_MAZE_TYPES)
        maze_name = '_'.join(['maze', 'type', str(maze_type), 'cnt', str(maze_type_cnt)])
    
        # generate mazes
        mazegen_cmd = repo_dir + '/mazegenerator/src/mazegen'
        mazegen_cmd += ' -o ' + maze_name
        mazegen_cmd += ' -m ' + str(maze_type)
        mazegen_proc = subprocess.Popen(mazegen_cmd.split(), stdout=subprocess.PIPE)
        output, error = mazegen_proc.communicate()

        # convert to ppm
        image = pyvips.Image.new_from_file(maze_name+'.svg', dpi=300)
        image.write_to_file(maze_name+'.ppm')

##############################################
############### TASK FUNCTIONS ###############
##############################################

if __name__ == "__main__":
    task_function()
