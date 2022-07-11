# procedurally generate 3D meshes of rooms using https://boschresearch.github.io/pcg_gazebo/
#
# For room generation example, see: 
#   https://boschresearch.github.io/pcg_gazebo/single_room_generation/
#   https://github.com/boschresearch/pcg_gazebo/blob/master/scripts/pcg-generate-sample-world-with-walls

import hydra
import numpy as np

from numpy.random import rand, randint
from hydra_zen import instantiate, make_config
from hydra.core.config_store import ConfigStore

_CONFIG_NAME = "pcg_rooms_app"

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

_DEFAULT_N_ROOMS = 10   # total number of rooms to create
_DEFAULT_N_CUBES_MAX = 50   # number of cubes to populate in each room
_DEFAULT_N_CYLINDERS_MAX = 50   # number of cubes to populate in each room
_DEFAULT_N_SPHERES_MAX = 50     # number of spheres to populate in each room

PCGRoomGenConfig = make_config(
    n_rooms = _DEFAULT_N_ROOMS,
    n_cubes_max = _DEFAULT_N_CUBES_MAX,
    n_cylinders_max = _DEFAULT_N_CYLINDERS_MAX,
    n_spheres_max = _DEFAULT_N_SPHERES_MAX )

# Store the top level config for command line interface
cs = ConfigStore.instance()
cs.store(_CONFIG_NAME, node=PCGRoomGenConfig)

##############################################
############### TASK FUNCTIONS ###############
##############################################

# non-configurable, fixed parameters
_WALL_THICKNESS = 0.5
_N_RECTANGLES_RANGE = (1, 17)
_N_POINTS_RANGE = (3, 33)
_WALL_HEIGHT_RANGE = (0.0, 20.0)

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: PCGRoomGenConfig):

    # instantiate the config object
    obj = instantiate(cfg)

    # iterate through each room to be generated
    for i in range(obj.n_rooms):

        # instantiate pcg generator command to be constructed
        pcg_cmd = "pcg-generate-sample-world-with-walls"

        # randomly select if room is points or merged rectangles
        # and randomize number of points/rectangles to use
        if rand() < 0.5:
            pcg_cmd += " --n-rectangles " + str(randint(*_N_RECTANGLES_RANGE))
        else:
            pcg_cmd += " --n-points " + str(randint(*_N_POINTS_RANGE))

        # add wall thickness command
        pcg_cmd += " --wall-thickness " + str(_WALL_THICKNESS)

        # randomize wall height
        wall_height = rand()*(_WALL_HEIGHT_RANGE[1] - _WALL_HEIGHT_RANGE[0]) + _WALL_HEIGHT_RANGE[0]
        pcg_cmd += " --wall-height " + str(wall_height)

        # get number of each shape to fill room
        pcg_cmd += " --n-cubes " + str(randint(obj.n_cubes_max))
        pcg_cmd += " --n-cylinders " + str(randint(obj.n_cylinders_max))
        pcg_cmd += " --n-spheres " + str(randint(obj.n_spheres_max))

        # randomize pitch and roll of shapes
        #TODO

        # manage export directories
        #TODO

        # call generation command
        print("DEBUG {}: {}".format(i, pcg_cmd))
        #TODO

        # convert stl wall meshes to obj
        #TODO


if __name__ == "__main__":
    task_function()


