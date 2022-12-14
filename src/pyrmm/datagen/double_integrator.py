import hydra
import multiprocessing
import time
import torch
import logging
import numpy as np
from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, instantiate, make_custom_builds_fn, builds
from hydra_zen import ZenField as zf

import pyrmm.utils.utils as U
from pyrmm.setups.double_integrator import DoubleIntegrator1DSetup, update_pickler_RealVectorStateSpace2
from pyrmm.datagen.sampler import sample_risk_metrics

_HASH_LEN = 5
_CONFIG_NAME = "doubleintegrator_datagen_app"
_SAVE_FNAME = U.format_save_filename(Path(__file__), _HASH_LEN)


##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# Dubins4dReachAvoid System Setup Config
_DEFAULT_POS_BOUNDS = [-10, 10]
_DEFAULT_VEL_BOUNDS = [-2, 2]
_DEFAULT_ACC_BOUNDS = [-1, 1]
DoubleIntegrator1DSetupConfig = pbuilds(DoubleIntegrator1DSetup,
    pos_bounds = _DEFAULT_POS_BOUNDS,
    vel_bounds = _DEFAULT_VEL_BOUNDS,
    acc_bounds = _DEFAULT_ACC_BOUNDS
)

# Default sampler and risk estimator configs
_DEFAULT_N_ENVIRONMENTS = 512
_DEFAULT_N_SAMPLES_PER_ENV = 32
_DEFAULT_DURATION = 2.0
_DEFAULT_N_BRANCHES = 32
_DEFAULT_TREE_DEPTH = 2
_DEFAULT_N_STEPS = 8
_DEFAULT_POLICY = 'uniform_random'
_DEFAULT_MAXTASKS = 16

# Top-level configuration and store for command line interface
make_config_input = {
    U.SYSTEM_SETUP: DoubleIntegrator1DSetupConfig,
    'n_environments': zf(int, _DEFAULT_N_ENVIRONMENTS),
    U.N_SAMPLES: zf(int, _DEFAULT_N_SAMPLES_PER_ENV),   # samples per environment
    U.DURATION: zf(float, _DEFAULT_DURATION),
    U.N_BRANCHES: zf(int, _DEFAULT_N_BRANCHES),
    U.TREE_DEPTH: zf(int,_DEFAULT_TREE_DEPTH),
    U.N_STEPS: zf(int,_DEFAULT_N_STEPS),
    U.POLICY: zf(str,_DEFAULT_POLICY),
    U.N_CORES: zf(int, multiprocessing.cpu_count()),
    'maxtasks': zf(int,_DEFAULT_MAXTASKS),
}
Config = make_config(**make_config_input)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################

# a logger for this file
log = logging.getLogger(__name__)

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config):
    '''Instantiate Double Integrator setup and generate risk metric data'''

    # instantiate config
    obj = instantiate(cfg)
    pos_bounds = getattr(cfg, U.SYSTEM_SETUP).pos_bounds
    # pos_bounds_range = pos_bounds[1] - pos_bounds[0]
    # pos_bounds_center = np.mean(pos_bounds)

    # update pickler to allow parallelization of ompl objects
    update_pickler_RealVectorStateSpace2()

    # iterate through each environment
    log.info("Starting DoubleIntegrator1D Risk Data Generation")
    t_start = time.time()
    for i in range(obj.n_environments):

        t_start_i = time.time()

        # randomize obstacle bounds
        rand_obst_bounds = np.random.uniform(*pos_bounds, 2)
        rand_obst_bounds = np.sort(rand_obst_bounds)

        # finish instantiating double integrator 1d setup object from 
        # partial object and random obstacle
        di1d_setup = getattr(obj, U.SYSTEM_SETUP)(obst_bounds=rand_obst_bounds)

        # create a unique name for saving risk metric data associated with environment
        sffx = "env_{}_of_{}".format(i+1, obj.n_environments)
        save_name = _SAVE_FNAME + '_' + sffx

        # sample states in environment and compute risk metrics
        log.info("Starting obstacle datagen: {}".format(sffx))
        risk_data = sample_risk_metrics(sysset=di1d_setup, cfg_obj=obj)
        torch.save(risk_data, open(save_name+".pt", "wb"))
        log.info(
            "Completed obstacle datagen: {}".format(sffx) +
            "\n---> elapsed time: {:.4f}".format(time.time() - t_start_i) + 
            "\n---> data samples: {}".format(getattr(obj, U.N_SAMPLES)) + 
            "\n---> data file: {}".format(save_name+".pt")
        )

    log.info(
            "DUBINS-4D RISK DATA GENERATION COMPLETE" +
            "\n---> Total Elapsed Time: {:.4f}".format(time.time() - t_start) + 
            "\n---> Total Obstacle Sets: {}".format(obj.n_environments) + 
            "\n---> Total Data Samples: {}".format(getattr(obj, U.N_SAMPLES)*obj.n_environments)
        )



if __name__ == "__main__":
    task_function()