import time
import hydra
import warnings
import logging
import numpy as np
import torch.optim as optim

from pathlib import Path
from typing import List
from pytorch_lightning import Trainer, seed_everything
from hydra.core.config_store import ConfigStore
from hydra_zen import make_custom_builds_fn, make_config, instantiate

import pyrmm.utils.utils as U
from pyrmm.setups.quadrotor import \
    ompl_to_numpy, update_pickler_quadrotorstate
from pyrmm.modelgen.data_modules import \
    BaseRiskMetricTrainingData, BaseRiskMetricDataModule
from pyrmm.modelgen.modules import \
    BaseRiskMetricModule, single_layer_nn_bounded_output


_CONFIG_NAME = "quadrotor_modelgen_app"

##############################################
### SYSTEM-SPECIFIC FUNCTSIONS AND CLASSES ###
##############################################

class QuadrotorPyBulletDataModule(BaseRiskMetricDataModule):
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

    def raw_data_to_numpy(self, raw_data:BaseRiskMetricTrainingData):
        '''convert raw data (e.g. OMPL objects) to numpy arrays'''

        # catch "ragged" array that would be caused by data with 
        # inconsistent observation sizes
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            state_samples = np.concatenate([ompl_to_numpy(s).reshape(1,13) for s in raw_data.state_samples], axis=0)
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

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

DataConf = pbuilds(QuadrotorPyBulletDataModule, 
    val_ratio=0.15, 
    batch_size=64, 
    num_workers=4,
    compile_verify_func=verify_compiled_data)

ModelConf = pbuilds(single_layer_nn_bounded_output, 
    num_neurons=64)

OptimConf = pbuilds(optim.Adam)

PLModuleConf = pbuilds(BaseRiskMetricModule, 
    model=ModelConf, 
    optimizer=OptimConf)

TrainerConf = pbuilds(Trainer, 
    max_epochs=2028, 
    precision=64, 
    reload_dataloaders_every_n_epochs=1, 
    progress_bar_refresh_rate=0)

ExperimentConfig = make_config(
    'train_data',
    test_data=None,
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

    # update pickler to enable parallelization of OMPL objects
    update_pickler_quadrotorstate()

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # compile all data in data directory
    datapaths = U.get_abs_pt_data_paths(obj.train_data)

    # finish instantiating data module
    data_module = obj.data_module(datapaths=datapaths)
    data_module.setup(stage='fit')
    hlog.info("training:n_data:{}".format(data_module.n_data))

    # extract the trained model input size from the observation data
    num_model_inputs = data_module.observation_shape[1]

    # finish instantiating the trainer
    trainer = obj.trainer()

    # finish instantiating pytorch lightning model and module
    pl_model = obj.pl_model(num_inputs=num_model_inputs)
    pl_module = obj.pl_module(num_inputs=num_model_inputs, model=pl_model)

    # train the model
    train_start_time = time.time()
    trainer.fit(pl_module, data_module)
    hlog.info("training:elapsed_time:{:.4f}".format(time.time()-train_start_time))
    for k, v in trainer.logged_metrics.items():
        hlog.info("trianing:metrics:{}:{}".format(k,v))

    if obj.test_data is not None:

        hlog.info("Starting testing and visualization")

        # compile all testing data
        test_datapaths = U.get_abs_pt_data_paths(obj.test_data)

        # finish instantiating data module
        # test_data_module = obj.data_module(datapaths=test_datapaths)
        test_data_module = QuadrotorPyBulletDataModule(
            datapaths=test_datapaths, 
            val_ratio=None, 
            batch_size=cfg.data_module.batch_size,
            num_workers=cfg.data_module.num_workers, 
            compile_verify_func=None)
        test_data_module.setup(stage='test')
        hlog.info("testing:n_data:{}".format(test_data_module.n_data))

        # run test data
        testing_start_time = time.time()
        trainer.test(ckpt_path="best", dataloaders=test_data_module.test_dataloader())
        hlog.info("testing:elapsed_time:{:.4f}".format(time.time()-testing_start_time))
        for k, v in trainer.logged_metrics.items():
            hlog.info("testing:metrics:{}:{}".format(k,v))

if __name__ == "__main__":
    task_function()
