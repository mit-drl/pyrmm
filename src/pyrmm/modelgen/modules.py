'''Pytorch Lightning modules for training risk metric models'''
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from typing import List
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, LightningModule
from hydra_zen.typing import Partial
from sklearn.preprocessing import MinMaxScaler

import pyrmm.utils.utils as U

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
        raw_data = dict()
        concat_data = []
        for dp in dpaths:
            cur_data = torch.load(dp)
            raw_data[str(dp)] = cur_data
            concat_data.extend(cur_data)
        n_data = len(concat_data)
        n_val = int(n_data*val_percent)
        n_train = n_data - n_val

        # convert SE2StateInternal objects into numpy arrays
        ssamples, rmetrics, lidars = tuple(zip(*concat_data))
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
        self.raw_data = raw_data
        # self.raw_data_paths = dpaths

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
        n_inputs: int,
        model: nn.Module,
        optimizer: Partial[optim.Adam],
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.example_input_array = torch.rand(32,n_inputs,dtype=torch.double)

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
