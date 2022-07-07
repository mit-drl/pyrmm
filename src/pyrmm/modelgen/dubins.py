import yaml
import torch
import hydra
import numpy as np
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from typing import List
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer, seed_everything, Callback
from hydra_zen import builds, make_custom_builds_fn, make_config, instantiate

import pyrmm.utils.utils as U
from pyrmm.modelgen.modules import RiskMetricDataModule, RiskMetricModule

_CONFIG_NAME = "dubins_modelgen_app"
_NUM_MODEL_INPUTS = 8

##############################################
################# MODEL DEF ##################
##############################################

def single_layer_nn(num_inputs: int, num_neurons: int) -> nn.Module:
    """y = sum(V sigmoid(X W + b))"""
    return nn.Sequential(
        nn.Linear(num_inputs, num_neurons),
        nn.Sigmoid(),
        nn.Linear(num_neurons, 1, bias=False),
    )

def linear_nn():
    return nn.Sequential(
        nn.Linear(3,1)
    )

class InputMonitor(Callback):
    '''Ref: https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch'''

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)


class CheckBatchGradient(Callback):
    '''Ref: https://www.pytorchlightning.ai/blog/3-simple-tricks-that-will-change-the-way-you-debug-pytorch'''
    
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()
        
        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)
        
        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")

def get_abs_data_paths(datadir):
    '''get list of absolute paths to .pt data fils in data_dir'''

    if datadir is None:
        raise Exception('please enter valid data directory')
    
    # get list of path objects to .pt files
    pathlist = list(Path(U.get_abs_path_str(datadir)).glob('*.pt'))
    
    # convert path objects to strings
    return [str(pth) for pth in pathlist]

def verify_hydrazen_rmm_data(datapaths: List):
    '''check that data compatibility
    Args:
        datapaths : list[PosixPath]
            list of paths to hydrazen outputs to be loaded
    '''

    for i, dp in enumerate(datapaths):
        cfg_path = dp.parent.joinpath('.hydra','config.yaml')
        with open(cfg_path, 'r') as cfg_file:
            cfg = yaml.full_load(cfg_file)
        
        if i == 0:
            # record system parameters
            # ppm_file = cfg[U.SYSTEM_SETUP]['ppm_file']
            speed = cfg[U.SYSTEM_SETUP]['speed']
            turn_rad = cfg[U.SYSTEM_SETUP]['min_turn_radius']

            # record risk metric estimation critical parameters
            dur = cfg[U.DURATION]
            depth = cfg[U.TREE_DEPTH]
            policy = cfg[U.POLICY]
            brnch = cfg[U.N_BRANCHES]
        
        else:
            # check system parameters match
            # assert ppm_file == cfg[U.SYSTEM_SETUP]['ppm_file']
            assert np.isclose(speed, cfg[U.SYSTEM_SETUP]['speed'])
            assert np.isclose(turn_rad, cfg[U.SYSTEM_SETUP]['min_turn_radius'])

            # check risk metric estimation critical parameters
            assert np.isclose(cfg[U.DURATION], dur)
            assert cfg[U.TREE_DEPTH] == depth
            assert cfg[U.POLICY] == policy
            assert cfg[U.N_BRANCHES] == brnch


##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

DataConf = pbuilds(RiskMetricDataModule, val_percent=0.15, batch_size=64, num_workers=4, data_verify_func=verify_hydrazen_rmm_data)

ModelConf = builds(single_layer_nn, num_inputs=_NUM_MODEL_INPUTS, num_neurons=64)

OptimConf = pbuilds(optim.Adam)

PLModuleConf = builds(RiskMetricModule, n_inputs=_NUM_MODEL_INPUTS, model=ModelConf, optimizer=OptimConf)

TrainerConf = pbuilds(Trainer, 
    max_epochs=2028, 
    precision=64, 
    reload_dataloaders_every_epoch=True, 
    progress_bar_refresh_rate=0)

ExperimentConfig = make_config(
    'datadir',
    data_module=DataConf,
    pl_module=PLModuleConf,
    trainer=TrainerConf,
    seed=1,
)

# Store the top level config for command line interface
cs = ConfigStore.instance()
cs.store(_CONFIG_NAME, node=ExperimentConfig)

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: ExperimentConfig):
    seed_everything(cfg.seed)

    # update pickler to enable multi-processing of OMPL objects
    U.update_pickler_se2stateinternal()

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # create data module
    datapaths = get_abs_data_paths(obj.datadir)
    data_module = obj.data_module(datapaths=datapaths)

    # select pseudo-test data and visualize
    pstest_dp = np.random.choice(list(data_module.raw_data.keys()))
    pstest_ssamples, pstest_rmetrics, pstest_lidars = tuple(zip(*data_module.raw_data[pstest_dp]))
    U.plot_dubins_data(Path(pstest_dp), desc="Truth", data=data_module.raw_data[pstest_dp])

    # finish instantiating the trainer
    trainer = obj.trainer(callbacks=[InputMonitor(), CheckBatchGradient()])
    # trainer = obj.trainer(callbacks=[InputMonitor()])

    # train the model
    trainer.fit(obj.pl_module, data_module)

    # randomly sample one of the datasets for pseudo-testing accuracy of model predictions
    # (i.e. not true testing because data is currently part of training)
    # convert SE2StateInternal objects into numpy arrays
    obj.pl_module.eval()
    pstest_lidars_np = np.asarray(pstest_lidars)
    pstest_lidars_scaled_pt = torch.from_numpy(data_module.observation_scaler.transform(pstest_lidars_np))
    pstest_pred_pt = obj.pl_module(pstest_lidars_scaled_pt)
    print('predicted data range: {} - {}'.format(torch.min(pstest_pred_pt), torch.max(pstest_pred_pt)))
    pstest_data = zip(pstest_ssamples, pstest_pred_pt.detach().numpy(), pstest_lidars)
    U.plot_dubins_data(Path(pstest_dp), desc='Inferred', data=pstest_data)


##############################################
############### TASK FUNCTIONS ###############
##############################################

if __name__ == "__main__":
    task_function()
