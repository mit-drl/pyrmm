import yaml
import torch
import hydra
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from typing import List
from hydra.core.config_store import ConfigStore
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything, Callback
from hydra_zen import builds, make_custom_builds_fn, make_config, instantiate
from hydra_zen.typing import Partial
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

import pyrmm.utils.utils as U

_CONFIG_NAME = "dubins_modelgen_app"
_NUM_MODEL_INPUTS = 8

##############################################
################# MODEL DEF ##################
##############################################

def se2_to_numpy(se2):
    '''convert OMPL SE2StateInternal object to numpy array'''
    return np.array([se2.getX(), se2.getY(), se2.getYaw()])

class RiskMetricDataModule(LightningDataModule):
    def __init__(self, datapaths: List, val_percent: float, batch_size: int, num_workers: int):
        '''loads data from torch save files
        Args:
            datapaths : list[str]
                list of path strings to hydrazen outputs to be loaded
            val_percent : float
                percent of data to be used 
            batch_size : int
                size of training batches
            num_workers : int
                number of workers to use for dataloader
        '''
        super().__init__()

        assert val_percent >= 0 and val_percent <= 1
        assert batch_size > 0

        self.batch_size = batch_size
        self.num_workers = num_workers

        # convert path strings in to absolute PosixPaths
        dpaths = [Path(dp).expanduser().resolve() for dp in datapaths]

        # ensure that all data is consistent on critical configs
        RiskMetricDataModule.verify_hydrazen_rmm_data(dpaths)

        # load data objects
        data = []
        for i, dp in enumerate(dpaths):
            data.extend(torch.load(dp))
        n_data = len(data)
        n_val = int(n_data*val_percent)
        n_train = n_data - n_val

        # convert SE2StateInternal objects into numpy arrays
        ssamples, rmetrics, lidars = tuple(zip(*data))
        ssamples_np = np.concatenate([se2_to_numpy(s).reshape(1,3) for s in ssamples], axis=0)
        rmetrics_np = np.asarray(rmetrics).reshape(-1,1)
        lidars_np = np.asarray(lidars)
        assert ssamples_np.shape[0] == rmetrics_np.shape[0] == lidars_np.shape[0]
        assert len(rmetrics_np.shape) == len(ssamples_np.shape) == len(lidars_np.shape)

        # Create input and output data regularizers
        # Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(ssamples_np)
        self.observation_scaler = MinMaxScaler()
        self.observation_scaler.fit(lidars_np)
        # self.output_scaler = MinMaxScaler()
        # self.output_scaler.fit(rmetrics_np)

        # scale and convert to tensor
        ssamples_scaled_pt = torch.from_numpy(self.state_scaler.transform(ssamples_np))
        lidars_scaled_pt = torch.from_numpy(self.observation_scaler.transform(lidars_np))
        rmetrics_pt = torch.from_numpy(rmetrics_np)
        
        # format into dataset
        # full_dataset = TensorDataset(ssamples_scaled_pt, rmetrics_pt)
        full_dataset = TensorDataset(lidars_scaled_pt, rmetrics_pt)

        # randomly split training and validation dataset
        self.train_dataset, self.val_dataset = random_split(full_dataset, [n_train, n_val])

        # store for later visualization use
        self.raw_data = data
        self.raw_data_paths = dpaths

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=len(self.val_dataset), shuffle=True)

    @staticmethod
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


class RiskMetricModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Partial[optim.Adam],
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.example_input_array = torch.rand(32,_NUM_MODEL_INPUTS,dtype=torch.double)

    def forward(self, inputs):
        return self.model(inputs)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # print('\nDEBUG: inputs shape: {}, targets shape {}\n'.format(inputs.shape, targets.shape))
        outputs = self.model(inputs)
        loss = F.mse_loss(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.print("Completed epoch {} of {}".format(self.current_epoch, self.trainer.max_epochs))
        self.log('avg_train_loss', avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        print('\n------------------------------\nSTARTING VALIDATION STEP\n')
        inputs, targets = batch
        pred = self.model(inputs)
        loss = F.mse_loss(pred, targets)
        self.print("\nvalidation loss:", loss.item())
        self.log('validation_loss', loss)

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

# def get_abs_path_str_list(pathlist):
#     '''get absolute paths for list of repo-relative paths '''
#     repo_dir = U.get_repo_path()
    
#     # if single string path is given, put it in 1-enty list
#     if isinstance(pathlist, str):
#         pathlist = [pathlist]
    
#     # if pathlist is something else (e.g. ListConfig), convert to list
#     if not isinstance(pathlist, list):
#         pathlist = list(pathlist)
        
#     return [str(Path(repo_dir).joinpath(dp)) for dp in pathlist]

def get_abs_path_str(rel_file_path):
    '''get absolute path of path relative to repo head'''
    repo_dir = U.get_repo_path()
    return str(Path(repo_dir).joinpath(rel_file_path))

def get_abs_data_paths(datadir):
    '''get list of absolute paths to .pt data fils in data_dir'''

    if datadir is None:
        raise Exception('please enter valid data directory')
    
    # get list of path objects to .pt files
    pathlist = list(Path(get_abs_path_str(datadir)).glob('*.pt'))
    
    # convert path objects to strings
    return [str(pth) for pth in pathlist]



##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# default_datapaths = [
#     # 'outputs/2022-01-25/15-41-18/datagen_dubins_0aa84_c8494.pt',
#     # 'outputs/2022-01-25/16-54-11/datagen_dubins_8299c_c8494.pt',
#     # 'outputs/2022-02-02/13-42-25/datagen_dubins_56d76_03af3.pt',
#     # 'outputs/2022-02-02/14-21-04/datagen_dubins_56d76_03af3.pt'
#     'outputs/2022-03-10/16-17-45/datagen_dubins_68efd_b9cc2.pt'
# ]
# DataPathConf = builds(get_data_paths, pathlist=default_datapaths)
# DataPathConf = pbuilds(get_abs_data_paths, datadir=None)

# DataConf = pbuilds(RiskMetricDataModule, datapaths=DataPathConf, val_percent=0.15, batch_size=64, num_workers=4)
DataConf = pbuilds(RiskMetricDataModule, val_percent=0.15, batch_size=64, num_workers=4)

ModelConf = builds(single_layer_nn, num_inputs=_NUM_MODEL_INPUTS, num_neurons=64)

OptimConf = pbuilds(optim.Adam)

PLModuleConf = builds(RiskMetricModule, model=ModelConf, optimizer=OptimConf)

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

    # visualize training data
    # U.plot_dubins_data(data_module.raw_data_paths[0], desc="Truth", data=data_module.raw_data)

    # finish instantiating the trainer
    trainer = obj.trainer(callbacks=[InputMonitor(), CheckBatchGradient()])
    # trainer = obj.trainer(callbacks=[InputMonitor()])

    # train the model
    trainer.fit(obj.pl_module, data_module)

    # randomly sample test data for visualization
    # convert SE2StateInternal objects into numpy arrays
    obj.pl_module.eval()
    # print('\n\nTEST EVALS\n\n')
    # test_inpt_np = np.array([
    #     [0,0,0], 
    #     [320, 0, 0], 
    #     [640, 0, 0], 
    #     [0, 200, 0],
    #     [0, 400, 0],
    #     [320, 200, 0],
    #     [320, 400, 0],
    #     [640, 200, 0],
    #     [640, 400, 0],
    #     [0,0,-np.pi], 
    #     [320, 0, -np.pi], 
    #     [640, 0, -np.pi], 
    #     [0, 200, -np.pi],
    #     [0, 400, -np.pi],
    #     [320, 200, -np.pi],
    #     [320, 400, -np.pi],
    #     [640, 200, -np.pi],
    #     [640, 400, -np.pi],
    #     [0,0,np.pi], 
    #     [320, 0, np.pi], 
    #     [640, 0, np.pi], 
    #     [0, 200, np.pi],
    #     [0, 400, np.pi],
    #     [320, 200, np.pi],
    #     [320, 400, np.pi],
    #     [640, 200, np.pi],
    #     [640, 400, np.pi],
    #     ])
    # t0_scaled_pt = torch.from_numpy(obj.data_module.input_scaler.transform(test_inpt_np))
    # t0_pred_pt = obj.pl_module(t0_scaled_pt)
    # for i, t in enumerate(test_inpt_np):
    #     print("orig state: {}\nscaled state: {}\nrisk pred: {}\n=================\n".format(t, t0_scaled_pt.numpy()[i], t0_pred_pt.detach().numpy()[i]))
    test_indices = np.random.choice(range(len(data_module.raw_data)), 10000)
    test_ssamples, test_rmetrics, test_lidars = tuple(zip(*data_module.raw_data))
    test_ssamples = np.array(test_ssamples)[test_indices]
    test_rmetrics = np.array(test_rmetrics)[test_indices]
    test_lidars = np.array(test_lidars)[test_indices]
    test_ssamples_np = np.concatenate([se2_to_numpy(s).reshape(1,3) for s in test_ssamples], axis=0)
    test_rmetrics_np = np.asarray(test_rmetrics)
    test_lidars_np = np.asarray(test_lidars)
    # test_ssamples_scaled_pt = torch.from_numpy(obj.data_module.input_scaler.transform(test_ssamples_np))
    test_lidars_scaled_pt = torch.from_numpy(data_module.observation_scaler.transform(test_lidars_np))
    test_rmetrics_pt = torch.tensor(test_rmetrics_np)
    test_pred_pt = obj.pl_module(test_lidars_scaled_pt)
    print('predicted data range: {} - {}'.format(torch.min(test_pred_pt), torch.max(test_pred_pt)))
    test_data = zip(test_ssamples, test_pred_pt.detach().numpy(), test_lidars)
    # U.plot_dubins_data(data_module.raw_data_paths[0], desc='Inferred', data=test_data)


##############################################
############### TASK FUNCTIONS ###############
##############################################

if __name__ == "__main__":
    task_function()
