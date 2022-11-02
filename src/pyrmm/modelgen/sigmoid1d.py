# A simple test case for training cbf-lrmm model
# Dynamics-less, 1D state space where risk value is sigmoid of state value
# and observation == state

import time
import torch
import hydra
import logging
import numpy as np
import torch.optim as optim

from pytorch_lightning import Trainer, seed_everything
from hydra_zen import builds, make_custom_builds_fn, make_config, instantiate
from hydra.core.config_store import ConfigStore

from pyrmm.modelgen.modules import \
    BaseRiskMetricTrainingData, \
    BaseRiskMetricDataModule, \
    ShallowRiskCBFPerceptron, \
    CBFLRMMModule

_CONFIG_NAME = "sigmoid_modelgen_app"

##############################################
############## DATA GENERATION ###############
##############################################

n_data = 100
state_samples = np.random.rand(n_data)*20-10
observations = state_samples
risk_metrics = torch.sigmoid(torch.tensor(state_samples)).numpy()
np_data = BaseRiskMetricTrainingData(
    state_samples=state_samples.reshape(-1,1),
    risk_metrics=risk_metrics.reshape(-1,1),
    observations=observations.reshape(-1,1))

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

DataConf = pbuilds(BaseRiskMetricDataModule, 
    val_ratio=0.15, 
    batch_size=64, 
    num_workers=4,
    compile_verify_func=None
)

ModelConf = pbuilds(ShallowRiskCBFPerceptron,  
    num_obs_inputs=1,
    num_state_features=1,
    num_neurons=64
)

OptimConf = pbuilds(optim.Adam)

PLModuleConf = pbuilds(CBFLRMMModule, 
    num_inputs = 1,
    optimizer=OptimConf
)

TrainerConf = pbuilds(Trainer, 
    max_epochs=2, 
    precision=64, 
    reload_dataloaders_every_n_epochs=1, 
)

ExperimentConfig = make_config(
    data_module=DataConf,
    pl_model=ModelConf,
    pl_module=PLModuleConf,
    trainer=TrainerConf,
    seed=1,
)

##############################################
############### TASK FUNCTIONS ###############
##############################################

# Store the top level config for command line interface
cs = ConfigStore.instance()
cs.store(_CONFIG_NAME, node=ExperimentConfig)

# a logger for this file managed by hydra
hlog = logging.getLogger(__name__)

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: ExperimentConfig):
    seed_everything(cfg.seed)

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # finish instantiating data module
    data_module = obj.data_module(datapaths=None)
    data_module.setup(stage='fit', np_data=np_data, store_raw_data=False)
    hlog.info("training:n_data:{}".format(data_module.n_data))

    # finish instantiating the trainer
    trainer = obj.trainer()

    # finish instantiating pytorch lightning model and module
    pl_model = obj.pl_model()
    pl_module = obj.pl_module(model=pl_model)

    # train the model
    train_start_time = time.time()
    trainer.fit(pl_module, data_module)
    hlog.info("training:elapsed_time:{:.4f}".format(time.time()-train_start_time))
    for k, v in trainer.logged_metrics.items():
        hlog.info("trianing:metrics:{}:{}".format(k,v))

if __name__ == "__main__":
    task_function()

