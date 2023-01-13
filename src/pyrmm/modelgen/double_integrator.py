# Hydrazen app to create and train a 1D Double-integrator model with 
# pre-existing data

import time
import hydra
import logging
import warnings
import numpy as np
import torch.optim as optim

from pathlib import Path
from typing import List
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer, seed_everything
from hydra_zen import builds, make_custom_builds_fn, make_config, instantiate

import pyrmm.utils.utils as U
from pyrmm.setups.double_integrator import \
    DoubleIntegrator1DSetup, update_pickler_RealVectorStateSpace2
from pyrmm.modelgen.data_modules import \
    BaseRiskMetricTrainingData, LSFORDataModule
from pyrmm.modelgen.modules import \
    ShallowRiskCBFPerceptron, CBFLRMMModule

_CONFIG_NAME = "doubleintegrator1d_modelgen_app"

#############################################
### SYSTEM-SPECIFIC FUNCTIONS AND CLASSES ###
#############################################

class DoubleIntegrator1DDataModule(LSFORDataModule):
    def __init__(self,
        datapaths: List[str],
        val_ratio: float, 
        batch_size: int, 
        num_workers: int,
        state_feature_map: callable,
        local_coord_map: callable,
        compile_verify_func: callable):

        super().__init__(
            datapaths=datapaths,
            val_ratio=val_ratio,
            batch_size=batch_size,
            num_workers=num_workers,
            state_feature_map=state_feature_map,
            local_coord_map=local_coord_map,
            compile_verify_func=compile_verify_func)

    def raw_data_to_numpy(self, raw_data:BaseRiskMetricTrainingData):
        '''convert raw data (e.g. OMPL objects) to numpy arrays
        
        Args:
            raw_data : BaseRiskMetricTrainingData
                pre-existing data that was created during the 
                datagen phase and loaded during the compile_raw_data execution
                which stores it in a BaseRiskMetricTrainingData namespace
                but still containing ompl objects
        
        Returns:
            BaseRiskMetricTrainingData
                pre-existing data converted out of its raw form (e.g. OMPL objects)
                into numpy arrays and packaged in a BaseRiskMetricTrainingData namespace
        '''

        # catch "ragged" array that would be caused by data with 
        # inconsistent observation sizes
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            state_samples= np.concatenate([DoubleIntegrator1DSetup.state_ompl_to_numpy(s).reshape(1,2) for s in raw_data.state_samples], axis=0)
            risk_metrics = np.asarray(raw_data.risk_metrics).reshape(-1,1)
            observations = np.asarray(raw_data.observations)

        return BaseRiskMetricTrainingData(
            state_samples= state_samples,
            risk_metrics = risk_metrics,
            observations = observations,
        )

def verify_compiled_data(datapaths: List[Path]):
    '''check that data compatibility (e.g. datagen params) for raw data
    Args:
        datapaths : List[Path]
            list of paths to hydrazen outputs to be loaded
    '''
    pass

def state_feature_map(state_sample):
    """trivial mapping from states to state feature vectors"""
    return state_sample

def local_coord_map(abs_state, ref_state):
    """conversion of euclidean state to local frame about ref_state

    Assumes no angular states that need to be handled
    """
    return abs_state-ref_state

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

DataConf = pbuilds(DoubleIntegrator1DDataModule, 
    val_ratio=0.15, 
    batch_size=64, 
    num_workers=4,
    state_feature_map=state_feature_map,
    local_coord_map=local_coord_map,
    compile_verify_func=verify_compiled_data
)

ModelConf = pbuilds(ShallowRiskCBFPerceptron,  
    # num_obs_inputs=2,
    num_state_features=2,
    num_neurons=8
)

OptimConf = pbuilds(optim.Adam)

PLModuleConf = pbuilds(CBFLRMMModule, 
    # num_inputs = 2,
    optimizer=OptimConf
)

TrainerConf = pbuilds(Trainer, 
    max_epochs=512, 
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
    update_pickler_RealVectorStateSpace2()

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # compile all training data in data directory
    datapaths = U.get_abs_pt_data_paths(obj.train_data)

    # finish instantiating data module
    data_module = obj.data_module(datapaths=datapaths)
    data_module.setup(stage='fit')
    hlog.info("training:n_data:{}".format(data_module.n_data))

    # extract the trained model input size from the observation data
    num_obs_inputs = data_module.observation_shape[1]

    # finish instantiating the trainer
    trainer = obj.trainer()

    # finish instantiating pytorch lightning model and module
    pl_model = obj.pl_model(num_obs_inputs=num_obs_inputs)
    pl_module = obj.pl_module(num_inputs=num_obs_inputs, model=pl_model)

    # train the model
    train_start_time = time.time()
    trainer.fit(pl_module, data_module)
    hlog.info("training:elapsed_time:{:.4f}".format(time.time()-train_start_time))
    for k, v in trainer.logged_metrics.items():
        hlog.info("trianing:metrics:{}:{}".format(k,v))

if __name__ == "__main__":
    task_function()