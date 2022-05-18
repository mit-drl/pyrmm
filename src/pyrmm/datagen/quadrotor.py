import hydra

from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, instantiate, builds

import pyrmm.utils.utils as U
from pyrmm.setups.quadrotor import QuadrotorPyBulletSetup


_CONFIG_NAME = "quadrotor_datagen_app"

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
    quadpb_setup = getattr(obj, U.SYSTEM_SETUP)()



