import time
import torch
import hydra
import logging
import warnings
import numpy as np
import torch.optim as optim

from pathlib import Path
from typing import List
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer, seed_everything, Callback
from hydra_zen import builds, make_custom_builds_fn, make_config, instantiate

import pyrmm.utils.utils as U
from pyrmm.setups.dubins4d import \
    Dubins4dReachAvoidSetup, \
    state_ompl_to_numpy, \
    update_pickler_dubins4dstate
from pyrmm.modelgen.modules import \
    RiskCtrlMetricDataModule, RiskCtrlMetricModule, \
    RiskCtrlMetricTrainingData, ShallowRiskCtrlMLP

_CONFIG_NAME = "dubins4d_modelgen_app"

#############################################
### SYSTEM-SPECIFIC FUNCTIONS AND CLASSES ###
#############################################

class Dubins4dReachAvoidDataModule(RiskCtrlMetricDataModule):
    def __init__(self,
        datapaths: List[str],
        val_ratio: float, 
        batch_size: int, 
        num_workers: int,
        compile_verify_func: callable):

        super().__init__(
            datapaths=datapaths,
            val_ratio=val_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            compile_verify_func=compile_verify_func)

    def raw_data_to_numpy(self, raw_data:RiskCtrlMetricTrainingData):
        '''convert raw data (e.g. OMPL objects) to numpy arrays'''

        # catch "ragged" array that would be caused by data with 
        # inconsistent observation sizes
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            state_samples= np.concatenate([state_ompl_to_numpy(s).reshape(1,4) for s in raw_data.state_samples], axis=0)
            risk_metrics = np.asarray(raw_data.risk_metrics).reshape(-1,1)
            observations = np.asarray(raw_data.observations)
            min_risk_ctrls = np.asarray(raw_data.min_risk_ctrls)
            min_risk_ctrl_durs = np.asarray(raw_data.min_risk_ctrl_durs).reshape(-1,1)

        return RiskCtrlMetricTrainingData(
            state_samples= state_samples,
            risk_metrics = risk_metrics,
            observations = observations,
            min_risk_ctrls=min_risk_ctrls,
            min_risk_ctrl_durs=min_risk_ctrl_durs
        )

def verify_compiled_data(datapaths: List[Path]):
    '''check that data compatibility (e.g. datagen params) for raw data
    Args:
        datapaths : List[Path]
            list of paths to hydrazen outputs to be loaded
    '''
    pass

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

DataConf = pbuilds(Dubins4dReachAvoidDataModule, 
    val_ratio=0.15, 
    batch_size=64, 
    num_workers=4,
    compile_verify_func=verify_compiled_data
)

ModelConf = pbuilds(ShallowRiskCtrlMLP,  
    num_neurons=64
)

OptimConf = pbuilds(optim.Adam)

PLModuleConf = pbuilds(RiskCtrlMetricModule,  
    optimizer=OptimConf
)

TrainerConf = pbuilds(Trainer, 
    max_epochs=2028, 
    precision=64, 
    reload_dataloaders_every_n_epochs=1, 
)

ExperimentConfig = make_config(
    "train_data",
    test_data=None,
    show_test_data=False,
    data_module=DataConf,
    pl_model=ModelConf,
    pl_module=PLModuleConf,
    trainer=TrainerConf,
    seed=1,
)

# Store the top level config for command line interface
cs = ConfigStore.instance()
cs.store(_CONFIG_NAME, node=ExperimentConfig)

##############################################
############### TASK FUNCTIONS ###############
##############################################

# a logger for this file managed by hydra
hlog = logging.getLogger(__name__)

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: ExperimentConfig):
    seed_everything(cfg.seed)

    # update pickler to enable multi-processing of OMPL objects
    update_pickler_dubins4dstate()

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # compile all training data in data directory
    datapaths = U.get_abs_pt_data_paths(obj.train_data)

    # finish instantiating data module
    data_module = obj.data_module(datapaths=datapaths)
    data_module.setup(stage='fit')
    hlog.info("training:n_data:{}".format(data_module.n_data))

    # extract the trained model input size from the observation data
    num_model_inputs = data_module.observation_shape[1]
    num_ctrl_dims = data_module.control_shape[1]

    # finish instantiating the trainer
    trainer = obj.trainer()

    # finish instantiating pytorch lightning model and module
    pl_model = obj.pl_model(num_inputs=num_model_inputs, num_ctrl_dims=num_ctrl_dims)
    pl_module = obj.pl_module(num_inputs=num_model_inputs, model=pl_model)

    # train the model
    train_start_time = time.time()
    trainer.fit(pl_module, data_module)
    hlog.info("training:elapsed_time:{:.4f}".format(time.time()-train_start_time))
    for k, v in trainer.logged_metrics.items():
        hlog.info("trianing:metrics:{}:{}".format(k,v))

if __name__ == "__main__":
    task_function()
