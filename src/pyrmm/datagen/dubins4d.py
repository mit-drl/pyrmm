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
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv
from pyrmm.setups.dubins4d import Dubins4dReachAvoidSetup, update_pickler_dubins4dstate
from pyrmm.datagen.sampler import sample_risk_metrics

_HASH_LEN = 5
_CONFIG_NAME = "dubins4d_datagen_app"
_SAVE_FNAME = U.format_save_filename(Path(__file__), _HASH_LEN)

##############################################
################# UTILITIES ##################
##############################################

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# Dubins4dReachAvoid System Setup Config
Dubins4dReachAvoidSetupConfig = pbuilds(Dubins4dReachAvoidSetup)

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
    # 'ppm_dir':'outputs/2022-03-10/19-31-52/',
    U.SYSTEM_SETUP: Dubins4dReachAvoidSetupConfig,
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
    '''Instantiate Dubins4d setup and generate risk metric data'''

    # instantiate config
    obj = instantiate(cfg)

    # update pickler to allow parallelization of ompl objects
    update_pickler_dubins4dstate()
    
    # iterate through each environment
    log.info("Starting Dubins4dReachAvoid Risk Data Generation")
    t_start = time.time()
    for i in range(obj.n_environments):

        t_start_i = time.time()

        # create environment with randomized obstacle
        env = Dubins4dReachAvoidEnv()

        # instantiate dubins4d reach-avoid setup with environment
        dubins4d_setup = getattr(obj, U.SYSTEM_SETUP)(env=env)

        # create a unique name for saving risk metric data associated with environment
        sffx = "env_{}_of_{}".format(i+1, obj.n_environments)
        save_name = _SAVE_FNAME + '_' + sffx

        # sample states in ppm config and compute risk metrics
        log.info("Starting obstacle datagen: {}".format(sffx))
        risk_data = sample_risk_metrics(sysset=dubins4d_setup, cfg_obj=obj)
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