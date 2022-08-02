import hydra
import multiprocess
import time
import torch
import logging
import numpy as np
from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, instantiate, make_custom_builds_fn, builds
from hydra_zen import ZenField as zf

import pyrmm.utils.utils as U
from pyrmm.setups.dubins import DubinsPPMSetup
from pyrmm.datagen.sampler import sample_risk_metrics

_HASH_LEN = 5
_CONFIG_NAME = "dubins_datagen_app"

##############################################
################# UTILITIES ##################
##############################################

_SAVE_FNAME = U.format_save_filename(Path(__file__), _HASH_LEN)

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(populate_full_signature=True)

_DEFAULT_SPEED = 10.0
_DEFAULT_MIN_TURN_RADIUS = 50.0
_DEFAULT_LIDAR_RESOLUTION = 1.0
_DEFAULT_LIDAR_NUM_RAYS = 8
DubinsPPMSetupConfig = builds(DubinsPPMSetup,
    speed = _DEFAULT_SPEED,
    min_turn_radius = _DEFAULT_MIN_TURN_RADIUS,
    lidar_resolution = _DEFAULT_LIDAR_RESOLUTION,
    lidar_n_rays = _DEFAULT_LIDAR_NUM_RAYS,
    zen_partial=True
)

# Default sampler and risk estimator configs
_DEFAULT_N_SAMPLES = 2048
_DEFAULT_DURATION = 2.0
_DEFAULT_N_BRANCHES = 32
_DEFAULT_TREE_DEPTH = 2
_DEFAULT_N_STEPS = 8
_DEFAULT_POLICY = 'uniform_random'
_DEFAULT_MAXTASKS = 16

# Top-level configuration and store for command line interface
make_config_input = {
    # 'ppm_dir':'outputs/2022-03-10/19-31-52/',
    U.SYSTEM_SETUP: DubinsPPMSetupConfig,
    U.N_SAMPLES: zf(int, _DEFAULT_N_SAMPLES),   # samples per environment
    U.DURATION: zf(float, _DEFAULT_DURATION),
    U.N_BRANCHES: zf(int, _DEFAULT_N_BRANCHES),
    U.TREE_DEPTH: zf(int,_DEFAULT_TREE_DEPTH),
    U.N_STEPS: zf(int,_DEFAULT_N_STEPS),
    U.POLICY: zf(str,_DEFAULT_POLICY),
    U.N_CORES: zf(int, multiprocess.cpu_count()),
    'maxtasks': zf(int,_DEFAULT_MAXTASKS),
}
Config = make_config('ppm_dir', **make_config_input)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################

# a logger for this file
log = logging.getLogger(__name__)

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config):
    '''Instantiate Dubins setup and generate risk metric data'''

    obj = instantiate(cfg)

    # update pickler to allow parallelization of ompl objects
    U.update_pickler_se2stateinternal()

    # get path to all ppm files in ppm_dir
    ppm_paths = list(Path(U.get_abs_path_str(obj.ppm_dir)).glob('*.ppm'))
    
    # iterate through each ppm configuration file for data generation
    log.info("Starting Dubins Risk Data Generation for Obstacle Sets: {}".format(obj.ppm_dir))
    t_start = time.time()
    for i, pp in enumerate(ppm_paths):

        t_start_i = time.time()

        # instantiate dubins ppm object from partial object and ppm file
        dubins_ppm_setup = getattr(obj, U.SYSTEM_SETUP)(ppm_file=str(pp))

        # create a unique name for saving risk metric data associated with specific ppm file
        save_name = _SAVE_FNAME + '_' + pp.stem

        # sample states in ppm config and compute risk metrics
        log.info("Starting obstacle set {} datagen ({} of {})".format(pp.name, i+1, len(ppm_paths)))
        risk_data = sample_risk_metrics(sysset=dubins_ppm_setup, cfg_obj=obj)
        torch.save(risk_data, open(save_name+".pt", "wb"))
        log.info(
            "Completed obstacle set {} ({} of {})".format(pp.name, i+1, len(ppm_paths)) +
            "\n---> elapsed time: {:.4f}".format(time.time() - t_start_i) + 
            "\n---> data samples: {}".format(getattr(obj, U.N_SAMPLES)) + 
            "\n---> data file: {}".format(save_name+".pt")
        )

    log.info(
            "DUBINS RISK DATA GENERATION COMPLETE" +
            "\n---> Total Elapsed Time: {:.4f}".format(time.time() - t_start) + 
            "\n---> Total Obstacle Sets: {}".format(len(ppm_paths)) + 
            "\n---> Total Data Samples: {}".format(getattr(obj, U.N_SAMPLES)*len(ppm_paths))
        )

if __name__ == "__main__":
    task_function()
