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
    CBFLRMMDataModule, \
    ShallowRiskCBFPerceptron, \
    CBFLRMMModule

##############################################
############## DATA GENERATION ###############
##############################################

def state_feature_map(state_sample):
    """trivial mapping from states to state feature vectors"""
    return state_sample

n_data = 2048
state_samples = np.random.rand(n_data)*20-10
observations = state_samples
risk_metrics = torch.sigmoid(torch.tensor(state_samples)).numpy()
np_data = BaseRiskMetricTrainingData(
    state_samples=state_samples.reshape(-1,1),
    risk_metrics=risk_metrics.reshape(-1,1),
    observations=observations.reshape(-1,1))


def task_function():

    # instantiate data module
    data_module = CBFLRMMDataModule(
        datapaths=None,
        val_ratio=0.15, 
        batch_size=64, 
        num_workers=4,
        state_feature_map=state_feature_map,
        compile_verify_func=None
    )
    data_module.setup(stage='fit', np_data=np_data, store_raw_data=False)

    # instantiate lightning module (i.e. model)
    model = ShallowRiskCBFPerceptron(
        num_obs_inputs=1,
        num_state_features=1,
        num_neurons=64
    )
    pl_module = CBFLRMMModule(
        num_inputs = 1,
        model = model,
        optimizer=optim.Adam
    )
    print("DEBUG: {}".format(pl_module.example_input_array))

    # instantiate trainer
    trainer = Trainer(
        max_epochs=512, 
        precision=64, 
        reload_dataloaders_every_n_epochs=1, 
    )

    # train the model
    train_start_time = time.time()
    trainer.fit(pl_module, data_module)
    # hlog.info("training:elapsed_time:{:.4f}".format(time.time()-train_start_time))
    # for k, v in trainer.logged_metrics.items():
    #     hlog.info("trianing:metrics:{}:{}".format(k,v))

if __name__ == "__main__":
    task_function()

