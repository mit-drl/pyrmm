# procedurally generate 3D meshes of rooms using https://boschresearch.github.io/pcg_gazebo/
#
# For room generation example, see: 
#   https://boschresearch.github.io/pcg_gazebo/single_room_generation/
#   https://github.com/boschresearch/pcg_gazebo/blob/master/scripts/pcg-generate-sample-world-with-walls

import hydra
import subprocess
import numpy as np
import xml.etree.ElementTree as ET

from pathlib import Path
from numpy.random import rand, randint
from hydra_zen import instantiate, make_config
from hydra.core.config_store import ConfigStore

import pyrmm.utils.utils as U

_CONFIG_NAME = "pcg_rooms_app"

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

# non-configurable, fixed parameters
_WALL_THICKNESS = 0.5
_INPUT_PREC = 5


_DEFAULT_N_ROOMS = 32   # total number of rooms to create
_DEFAULT_N_RECTANGLES_MIN = 1   # number of rectangles used to create rooms
_DEFAULT_N_RECTANGLES_MAX = 32   # number of rectangles used to create rooms
_DEFAULT_X_ROOM_RANGE_MIN = 5   # length scale of room in x-direction
_DEFAULT_X_ROOM_RANGE_MAX = 50   # length scale of room in y-direction
_DEFAULT_Y_ROOM_RANGE_MIN = 5   # length scale of room in x-direction
_DEFAULT_Y_ROOM_RANGE_MAX = 50   # length scale of room in y-direction
_DEFAULT_WALL_HEIGHT_MIN = 0.1  # minimum wall height
_DEFAULT_WALL_HEIGHT_MAX = 5.0  # maximum wall height
_DEFAULT_N_CUBES_MIN = 0    # number of cubes to populate in each room
_DEFAULT_N_CUBES_MAX = 50   # number of cubes to populate in each room
_DEFAULT_N_CYLINDERS_MIN = 0   # number of cubes to populate in each room
_DEFAULT_N_CYLINDERS_MAX = 50   # number of cubes to populate in each room
_DEFAULT_N_SPHERES_MIN = 0     # number of spheres to populate in each room
_DEFAULT_N_SPHERES_MAX = 50     # number of spheres to populate in each room

PCGRoomGenConfig = make_config(
    n_rooms = _DEFAULT_N_ROOMS,
    n_rectangles_min = _DEFAULT_N_RECTANGLES_MIN,
    n_rectangles_max = _DEFAULT_N_RECTANGLES_MAX,
    x_room_range_min = _DEFAULT_X_ROOM_RANGE_MIN,
    x_room_range_max = _DEFAULT_X_ROOM_RANGE_MAX,
    y_room_range_min = _DEFAULT_Y_ROOM_RANGE_MIN,
    y_room_range_max = _DEFAULT_Y_ROOM_RANGE_MAX,
    wall_height_min = _DEFAULT_WALL_HEIGHT_MIN,
    wall_height_max = _DEFAULT_WALL_HEIGHT_MAX,
    n_cubes_min = _DEFAULT_N_CUBES_MIN,
    n_cubes_max = _DEFAULT_N_CUBES_MAX,
    n_cylinders_min = _DEFAULT_N_CYLINDERS_MIN,
    n_cylinders_max = _DEFAULT_N_CYLINDERS_MAX,
    n_spheres_min = _DEFAULT_N_SPHERES_MIN,
    n_spheres_max = _DEFAULT_N_SPHERES_MAX )

# Store the top level config for command line interface
cs = ConfigStore.instance()
cs.store(_CONFIG_NAME, node=PCGRoomGenConfig)

##############################################
############### TASK FUNCTIONS ###############
##############################################

def overwrite_stl_in_model_sdf(modelsdf_filepath:str):
    '''open model.sdf and overwrite the .stl uri to .obj'''

    # parse xml tree from sdf file
    sdf_tree = ET.parse(modelsdf_filepath)
    sdf_root = sdf_tree.getroot()

    # update the uri elements to point to obj files instead of stl
    uri_prefix = 'file://'
    for uri in sdf_root.iter('uri'):

        # get uri string and strip the file:// prefix 
        # for ease of using with pathlib.Path
        stl_uri = uri.text
        assert stl_uri.startswith(uri_prefix)
        stl_path = Path(stl_uri[len(uri_prefix):])

        # overwrite the text on the uri tag
        # ensuring that you add back the uri file:// formatiing
        if stl_path.suffix == '.stl':
            uri.text = str(stl_path.with_suffix('.obj').as_uri())

    # write the modified xml file back to the same sdf file
    sdf_tree.write(modelsdf_filepath)


def pcg_world_to_pybullet_sdf(dirpath:str, world_name:str, debug:bool=False):
    '''makes the pcg_gazebo generated room compatible with pybullet
    
    Args:
        dir_path : str
            absolute path to directory containing pcg-generated room .world
        world_name : str
            prefix of room's .world file to be converted
    '''

    # reconstruct paths and filenames and assert existence and type
    abs_dirpath = Path(dirpath).expanduser().resolve()
    assert abs_dirpath.exists()
    assert abs_dirpath.is_dir()

    abs_world_filepath = abs_dirpath.joinpath(world_name + '.world')
    assert abs_world_filepath.exists()
    assert abs_world_filepath.is_file()

    abs_walls_dirpath = abs_dirpath.joinpath(world_name + '_walls')
    assert abs_walls_dirpath.exists()
    assert abs_walls_dirpath.is_dir()

    abs_model_filepath = abs_walls_dirpath.joinpath('model.sdf')
    assert abs_model_filepath.exists()
    assert abs_model_filepath.is_file()

    abs_meshes_dirpath = abs_walls_dirpath.joinpath('meshes')
    assert abs_walls_dirpath.exists()
    assert abs_walls_dirpath.is_dir()

    # get list of all wall mesh stl files
    abs_mesh_stl_filepaths = abs_meshes_dirpath.glob('*.stl')

    # call blender conversion script to convert stl to obj
    blender_cmd_part = 'blender -b'
    blender_cmd_part += ' --python ' + U.get_abs_path_str('src/pyrmm/utils/blender_convert_stl_to_obj.py')
    blender_cmd_part += ' -- '
    vhacd_cmd_part = U.get_repo_path() + '/v-hacd/app/TestVHACD ' 
    for stl_file in abs_mesh_stl_filepaths:
        obj_file = stl_file.with_suffix('.obj')
        blender_cmd = blender_cmd_part + str(stl_file) + ' ' + str(obj_file)
        if debug:
            print("DEBUG: Blender conversion command:\n", blender_cmd)
        subprocess.run(blender_cmd.split())
        
        # check that .obj file was created
        assert obj_file.exists()
        assert obj_file.is_file()

        # create convex decomposition of walls
        vhacd_cmd = vhacd_cmd_part + str(obj_file)
        subprocess.run(vhacd_cmd.split())

        # overwrite old, non decomposed obj file
        print("Overwriting {} with decomp.obj and removing decomp.stl".format(obj_file.name))
        decomp_file = Path.cwd().joinpath('decomp.obj')
        decomp_file.replace(obj_file)
        Path.cwd().joinpath('decomp.stl').unlink()

        # check that .obj file was created
        assert obj_file.exists()
        assert obj_file.is_file()


    # modify walls model.sdf to point to obj instead of stl
    overwrite_stl_in_model_sdf(str(abs_model_filepath))

def check_config_inputs(cfg: PCGRoomGenConfig):
    '''make sure config inputs are logical'''
    assert cfg.n_rooms > 0
    assert cfg.n_rectangles_min > 0
    assert cfg.n_rectangles_max >= cfg.n_rectangles_min
    assert cfg.x_room_range_min > 0
    assert cfg.x_room_range_max >= cfg.x_room_range_min
    assert cfg.y_room_range_min > 0
    assert cfg.y_room_range_max >= cfg.y_room_range_min
    assert cfg.wall_height_min > 0
    assert cfg.wall_height_max >= cfg.wall_height_min
    assert cfg.n_cubes_min >= 0
    assert cfg.n_cubes_max >= cfg.n_cubes_min
    assert cfg.n_cylinders_min >= 0
    assert cfg.n_cylinders_max >= cfg.n_cylinders_min
    assert cfg.n_spheres_min >= 0
    assert cfg.n_spheres_max >= cfg.n_spheres_min

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: PCGRoomGenConfig):

    # check inputs
    check_config_inputs(cfg=cfg)

    # instantiate the config object
    obj = instantiate(cfg)

    # iterate through each room to be generated
    for i in range(obj.n_rooms):

        status_str = '{}/{}'.format(i+1,obj.n_rooms)
        print(
            '\n----------------------\n' +
            'GENERATING ROOM ' + status_str + 
            '\n----------------------\n')

        # instantiate pcg generator command to be constructed
        pcg_cmd = "pcg-generate-sample-world-with-walls"

        # randomly generate number of rectangles to merge to make rooms
        pcg_cmd += " --n-rectangles " + str(randint(obj.n_rectangles_min, obj.n_rectangles_max+1))

        # radnomly generate dimensions of rooms
        x_range = rand()*(obj.x_room_range_max - obj.x_room_range_min) + obj.x_room_range_min
        y_range = rand()*(obj.y_room_range_max - obj.y_room_range_min) + obj.y_room_range_min
        pcg_cmd += " --x-room-range " + str(x_range)[:_INPUT_PREC]
        pcg_cmd += " --y-room-range " + str(y_range)[:_INPUT_PREC]

        # add wall thickness command
        pcg_cmd += " --wall-thickness " + str(_WALL_THICKNESS)

        # randomize wall height
        wall_height = rand()*(obj.wall_height_max - obj.wall_height_min) + obj.wall_height_min
        pcg_cmd += " --wall-height " + str(wall_height)[:_INPUT_PREC]

        # get number of each shape to fill room
        pcg_cmd += " --n-cubes " + str(randint(obj.n_cubes_min, obj.n_cubes_max+1))
        pcg_cmd += " --n-cylinders " + str(randint(obj.n_cylinders_min, obj.n_cylinders_max+1))
        pcg_cmd += " --n-spheres " + str(randint(obj.n_spheres_min, obj.n_spheres_max+1))

        # randomize pitch and roll of shapes
        pcg_cmd += " --set-random-roll --set-random-pitch"

        # manage export directories
        world_name = "pcg_room_" + str(i).zfill(3)
        pcg_cmd += " --export-world-dir ./ --export-models-dir ./"
        pcg_cmd += " --world-name " + world_name

        # call generation command
        print(status_str,": pcg-gazebo room generation command:\n", pcg_cmd)
        pcg_proc = subprocess.Popen(pcg_cmd.split(), stdout=subprocess.PIPE)
        output, error = pcg_proc.communicate()

        # convert stl wall meshes to obj
        pcg_world_to_pybullet_sdf(dirpath="./", world_name=world_name, debug=True)

        print(
            '\n----------------------\n',
            'ROOM {} COMPLETE'.format(status_str),
            '\n----------------------\n',)

        print(
            '\n---------------------------------------------------\n'+
            '---------------------------------------------------\n'+
            '---------------------------------------------------\n'
            )


if __name__ == "__main__":
    task_function()


