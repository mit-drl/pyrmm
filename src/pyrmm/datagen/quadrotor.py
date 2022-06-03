import hydra
import multiprocess
import time
import torch
import numpy as np
import pybullet as pb
import pybullet_data as pbd

from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, instantiate, builds
from hydra_zen import ZenField as zf

import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD
from pyrmm.datagen.sampler import sample_risk_metrics
from pyrmm.setups.quadrotor import QuadrotorPyBulletSetup


_HASH_LEN = 5
_CONFIG_NAME = "quadrotor_datagen_app"

_SAVE_FNAME = U.format_save_filename(Path(__file__), _HASH_LEN)

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

_DEFAULT_LIDAR_RANGE = 100.0
_DEFAULT_LIDAR_ANGLES = [
        (np.pi/2, 0),
        (np.pi/2, np.pi/2),
        (np.pi/2, np.pi),
        (np.pi/2, 3*np.pi/2),
        (0,0),
        (np.pi,0)
    ]
QuadrotorPyBulletSetupConfig = builds(QuadrotorPyBulletSetup,
    lidar_range = _DEFAULT_LIDAR_RANGE,
    lidar_angles = _DEFAULT_LIDAR_ANGLES
)

# Default sampler and risk estimator configs
_DEFAULT_N_SAMPLES = 2048
_DEFAULT_DURATION = 2.0
_DEFAULT_N_BRANCHES = 32
_DEFAULT_TREE_DEPTH = 2
_DEFAULT_N_STEPS = 8
_DEFAULT_POLICY = 'uniform_random'
_DEFAULT_MAXTASKS = 16

make_config_input = {
    U.SYSTEM_SETUP: QuadrotorPyBulletSetupConfig,
    U.N_SAMPLES: zf(int, _DEFAULT_N_SAMPLES),
    U.DURATION: zf(float, _DEFAULT_DURATION),
    U.N_BRANCHES: zf(int, _DEFAULT_N_BRANCHES),
    U.TREE_DEPTH: zf(int,_DEFAULT_TREE_DEPTH),
    U.N_STEPS: zf(int,_DEFAULT_N_STEPS),
    U.POLICY: zf(str,_DEFAULT_POLICY),
    U.N_CORES: zf(int, multiprocess.cpu_count()),
    'maxtasks': zf(int,_DEFAULT_MAXTASKS),
}
Config = make_config(**make_config_input)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################

def create_setup_and_sample_risk_metrics(ss_cfg):
    '''functionalized SystemSetup creation and risk metric eval for multiprocessing
    Args:
        ss_cfg : 
            SystemSetup Config to instantiate SystemSetup
    '''

    # instantiate config object to create system setup object
    obj = instantiate(ss_cfg)

    # instantiate quadrotor pybullet setup object
    quadpb_setup = getattr(obj, U.SYSTEM_SETUP)

    # load environment URDF
    pb.setAdditionalSearchPath(pbd.getDataPath())
    bld_body_id = pb.loadURDF("samurai.urdf")

    # sample states in environment and compute risk metrics
    # Note: don't run sample_risk_metric in multiprocess mode
    # since multi-processing is brokered in outer loop when using pybullet
    risk_data = sample_risk_metrics(sysset=quadpb_setup, cfg_obj=obj, multiproc=False)
    return risk_data


@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config): 
    ''' Instantiate QuadrotorPyBullet setup and sample risk metric data'''

    # update pickler to enable parallelization of OMPL objects
    QD.update_pickler_quadrotorstate()

    # call helper function to instantiate quad system setup and get risk data
    t_start = time.time()
    risk_data = create_setup_and_sample_risk_metrics(ss_cfg=cfg)
    torch.save(risk_data, open(_SAVE_FNAME+".pt", "wb"))
    print("\nTotal elapsed time: {:.2f}".format(time.time()-t_start))


if __name__ == "__main__":
    task_function()




