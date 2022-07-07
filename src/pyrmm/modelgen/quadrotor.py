import hydra
import numpy as np

from pytorch_lightning import Trainer, seed_everything
from hydra.core.config_store import ConfigStore
from hydra_zen import make_custom_builds_fn, make_config, instantiate

import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD
from pyrmm.setups.quadrotor import ompl_to_numpy, update_pickler_quadrotorstate
from pyrmm.modelgen.modules import RiskMetricDataModule, RiskMetricModule, \
    compile_state_risk_obs_data


_CONFIG_NAME = "quadrotor_modelgen_app"

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

DataConf = pbuilds(RiskMetricDataModule, val_percent=0.15, batch_size=64, num_workers=4)

ExperimentConfig = make_config(
    'datadir',
    data_module=DataConf,
    seed=1,
)

# Store the top level config for command line interface
cs = ConfigStore.instance()
cs.store(_CONFIG_NAME, node=ExperimentConfig)

##############################################
############### TASK FUNCTIONS ###############
##############################################

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: ExperimentConfig):
    seed_everything(cfg.seed)

    # update pickler to enable parallelization of OMPL objects
    update_pickler_quadrotorstate()

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # compile all data in data directory
    datapaths = U.get_abs_pt_data_paths(obj.datadir)
    compiled_data = compile_state_risk_obs_data(
        datapaths=datapaths,
        data_verify_func=None)

    # convert states, risks, and observations into numpy arrays
    state_samples_np = np.concatenate([ompl_to_numpy(s).reshape(1,13) for s in compiled_data[0]], axis=0)
    risk_metrics_np = np.asarray(compiled_data[1]).reshape(-1,1)
    observations_np = np.asarray(compiled_data[2])
    print("DEBUG: state samples shape:{}\nrisk metrics shape: {}\n observations shape:{}".format(
        state_samples_np.shape, risk_metrics_np.shape, observations_np.shape
    ))

    # create data module
    data_module = obj.data_module(
        state_samples_np=state_samples_np,
        risk_metrics_np=risk_metrics_np,
        observations_np=observations_np)

if __name__ == "__main__":
    task_function()
