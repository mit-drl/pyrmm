import hydra

from pytorch_lightning import Trainer, seed_everything
from hydra_zen import make_custom_builds_fn, make_config, instantiate

import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD
from pyrmm.modelgen.modules import RiskMetricDataModule, RiskMetricModule


_CONFIG_NAME = "quadrotor_modelgen_app"

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

DataConf = pbuilds(RiskMetricDataModule, val_percent=0.15, batch_size=64, num_workers=4, data_verify_func=None)

ExperimentConfig = make_config(
    'datadir',
    data_module=DataConf,
    seed=1,
)

##############################################
############### TASK FUNCTIONS ###############
##############################################

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: ExperimentConfig):
    seed_everything(cfg.seed)

    # update pickler to enable parallelization of OMPL objects
    QD.update_pickler_quadrotorstate()

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # create data module
    datapaths = U.get_abs_pt_data_paths(obj.datadir)
    data_module = obj.data_module(datapaths=datapaths)
