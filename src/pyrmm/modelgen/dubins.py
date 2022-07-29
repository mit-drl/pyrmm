import yaml
import time
import torch
import hydra
import warnings
import numpy as np
import torch.optim as optim

from pathlib import Path
from typing import List
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer, seed_everything, Callback
from hydra_zen import builds, make_custom_builds_fn, make_config, instantiate

import pyrmm.utils.utils as U
from pyrmm.modelgen.modules import RiskMetricDataModule, RiskMetricModule, \
    RiskMetricTrainingData, single_layer_nn_bounded_output

_CONFIG_NAME = "dubins_modelgen_app"
# _NUM_MODEL_INPUTS = 8

##############################################
### SYSTEM-SPECIFIC FUNCTSIONS AND CLASSES ###
##############################################

def se2_to_numpy(se2):
    '''convert OMPL SE2StateInternal object to numpy array'''
    return np.array([se2.getX(), se2.getY(), se2.getYaw()])

class DubinsPPMDataModule(RiskMetricDataModule):
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

    def raw_data_to_numpy(self, raw_data:RiskMetricTrainingData):
        '''convert raw data (e.g. OMPL objects) to numpy arrays'''

        # catch "ragged" array that would be caused by data with 
        # inconsistent observation sizes
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            state_samples= np.concatenate([se2_to_numpy(s).reshape(1,3) for s in raw_data.state_samples], axis=0)
            risk_metrics = np.asarray(raw_data.risk_metrics).reshape(-1,1)
            observations = np.asarray(raw_data.observations)

        return RiskMetricTrainingData(
            state_samples= state_samples,
            risk_metrics = risk_metrics,
            observations = observations,
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

def verify_hydrazen_rmm_data(datapaths: List[Path]):
    '''check that data compatibility (e.g. datagen params) for raw data
    Args:
        datapaths : List[Path]
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

DataConf = pbuilds(DubinsPPMDataModule, 
    val_ratio=0.15, 
    batch_size=64, 
    num_workers=4,
    compile_verify_func=verify_hydrazen_rmm_data
    )

ModelConf = pbuilds(single_layer_nn_bounded_output,  
    num_neurons=64)

OptimConf = pbuilds(optim.Adam)

PLModuleConf = pbuilds(RiskMetricModule,  
    # model=ModelConf, 
    optimizer=OptimConf)

TrainerConf = pbuilds(Trainer, 
    max_epochs=2028, 
    precision=64, 
    reload_dataloaders_every_n_epochs=1, 
    progress_bar_refresh_rate=0)

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

@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: ExperimentConfig):
    seed_everything(cfg.seed)

    # update pickler to enable multi-processing of OMPL objects
    U.update_pickler_se2stateinternal()

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # instantiate results summary
    result_summary = dict()
    result_summary['training'] = dict()

    # compile all training data in data directory
    datapaths = U.get_abs_pt_data_paths(obj.train_data)

    # finish instantiating data module
    data_module = obj.data_module(datapaths=datapaths)
    data_module.setup(stage='fit')
    result_summary['training']['n_data'] = data_module.n_data

    # extract the trained model input size from the observation data
    num_model_inputs = data_module.observation_shape[1]

    # finish instantiating the trainer
    trainer = obj.trainer(callbacks=[InputMonitor(), CheckBatchGradient()])

    # finish instantiating pytorch lightning model and module
    pl_model = obj.pl_model(num_inputs=num_model_inputs)
    pl_module = obj.pl_module(num_inputs=num_model_inputs, model=pl_model)

    # train the model
    train_start_time = time.time()
    trainer.fit(pl_module, data_module)
    result_summary['training']['elapsed_time'] = time.time() - train_start_time
    result_summary['training']['metrics'] = dict()
    for k, v in trainer.logged_metrics.items():
        if k.startswith('val'):
            result_summary['training']['metrics'][k] = v

    if obj.test_data is not None:

        print('\n\n~~~~TESTING AND VISUALIZATION~~~~\n\n')

        result_summary['testing'] = dict()

        # compile all testing data
        test_datapaths = U.get_abs_pt_data_paths(obj.test_data)

        # finish instantiating data module
        # test_data_module = obj.data_module(datapaths=test_datapaths)
        test_data_module = DubinsPPMDataModule(
            datapaths=test_datapaths, 
            val_ratio=None, 
            batch_size=cfg.data_module.batch_size,
            num_workers=cfg.data_module.num_workers, 
            compile_verify_func=None)
        test_data_module.setup(stage='test')
        result_summary['testing']['n_data'] = test_data_module.n_data

        # run test data
        testing_start_time = time.time()
        trainer.test(ckpt_path="best", dataloaders=test_data_module.test_dataloader())
        result_summary['testing']['elapsed_time'] = time.time() - testing_start_time
        result_summary['testing']['metrics'] = trainer.logged_metrics

        # Visualize test data
        separated_raw_test_data = test_data_module.separated_raw_data
        test_dp = np.random.choice(list(separated_raw_test_data.keys()))
        test_ssamples, test_rmetrics, test_observations = tuple(zip(*separated_raw_test_data[test_dp]))
        U.plot_dubins_data(Path(test_dp), desc="Truth", data=separated_raw_test_data[test_dp], show=obj.show_test_data)

        # Evaluate model on test data and visualize
        pl_module.eval()
        test_observations_np = np.asarray(test_observations)
        test_observations_scaled_pt = torch.from_numpy(data_module.observation_scaler.transform(test_observations_np))
        test_pred_rmetrics_pt = pl_module(test_observations_scaled_pt)
        test_targ_rmetrics_pt = torch.from_numpy(np.asarray(test_rmetrics)).unsqueeze(1)
        assert test_pred_rmetrics_pt.shape == test_targ_rmetrics_pt.shape
        print('predicted data range: {} - {}'.format(torch.min(test_pred_rmetrics_pt), torch.max(test_pred_rmetrics_pt)))
        print('maximum absolute risk metric error: {}'.format(torch.max(torch.abs(test_pred_rmetrics_pt - test_targ_rmetrics_pt))))
        test_full_data = zip(test_ssamples, test_pred_rmetrics_pt.detach().numpy(), test_observations)
        U.plot_dubins_data(Path(test_dp), desc='Inferred', data=test_full_data, show=obj.show_test_data)

    # write result summary to file
    with open('result_summary.yaml', 'w') as result_file:
        yaml.dump(result_summary, result_file, default_flow_style=False)


if __name__ == "__main__":
    task_function()
