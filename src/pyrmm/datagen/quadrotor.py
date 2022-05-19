import hydra
import time
import pybullet as pb
import pybullet_data as pbd

from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, instantiate, builds

import pyrmm.utils.utils as U
from pyrmm.datagen.sampler import sample_risk_metrics
from pyrmm.setups.quadrotor import QuadrotorPyBulletSetup


_HASH_LEN = 5
_CONFIG_NAME = "quadrotor_datagen_app"

_SAVE_FNAME = U.format_save_filename(Path(__file__), _HASH_LEN)

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

QuadrotorPyBulletSetupConfig = builds(QuadrotorPyBulletSetup)

make_config_input = {
    U.SYSTEM_SETUP: QuadrotorPyBulletSetupConfig,
}
Config = make_config(**make_config_input)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config): 
    ''' Instantiate QuadrotorPyBullet setup and sample risk metric data'''

    obj = instantiate(cfg)

    # update pickler to enable parallelization of OMPL objects
    # TODO

    # instantiate quadrotor pybullet setup object
    quadpb_setup = getattr(obj, U.SYSTEM_SETUP)

    # load environment URDF
    pb.setAdditionalSearchPath(pbd.getDataPath())
    bld_body_id = pb.loadURDF("samurai.urdf")

    # sample states in environment and compute risk metrics
    t_start = time.time()
    sample_risk_metrics(sysset=quadpb_setup, cfg_obj=obj, save_name=_SAVE_FNAME)
    print("\nTotal elapsed time: {:.2f}".format(time.time()-t_start))


if __name__ == "__main__":
    task_function()




