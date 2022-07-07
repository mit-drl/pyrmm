'''Pytorch Lightning modules for training risk metric models'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from typing import List, Tuple
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, LightningModule
from hydra_zen.typing import Partial
from sklearn.preprocessing import MinMaxScaler

import pyrmm.utils.utils as U

def compile_state_risk_obs_data(datapaths, data_verify_func:callable=None):
    ''' extract and concat state samples, risk metrics, and observations from data files

    Assumes that data from datagen has been saved in state-risk-observation ordered format

    Args:
        datapaths : list[str]
            list of path strings to hydrazen outputs to be loaded
        data_verify_func : callable
                function to call to verify consistency of data
    
    Returns:
        state_samples : tuple(state)
            state samples in their native format (e.g. OMPL objects)
        risk_metrics : tuple(state)
            risk metric evaluated at corresponding sampled state
        observations : tuple(array)
            observeState outputs at each corresponding sampled state
        raw_data : dict
            raw data separated by data files, useful for visualization
    '''
    # convert path strings in to absolute PosixPaths
    dpaths = [Path(dp).expanduser().resolve() for dp in datapaths]

    # ensure that all data is consistent on critical configs
    if data_verify_func is not None:
        data_verify_func(dpaths)

    # load data objects
    raw_data = dict()
    concat_data = []
    for dp in dpaths:
        cur_data = torch.load(dp)
        raw_data[str(dp)] = cur_data
        concat_data.extend(cur_data)

    # zip data into three immutable, indice-aligned objects
    state_samples, risk_metrics, observations = tuple(zip(*concat_data))

    return state_samples, risk_metrics, observations, raw_data


class RiskMetricDataModule(LightningDataModule):
    def __init__(self,
        state_samples_np: Tuple[np.ndarray],
        risk_metrics_np: Tuple[float],
        observations_np: Tuple[np.ndarray], 
        validation_percent: float, 
        batch_size: int, 
        num_workers: int):
        '''loads data from torch save files
        Args:
            state_samples_np : tuple(np.ndarray)
                tuple (immutable) of state samples formatted into numpy arrays. 
                i-th element is i-th sampled state represented as a numpy array
            risk_metrics_np : tuple(float)
                tuple (immutable) of risk metric as a floating point value. 
                i-th element is risk metric evaluated at the i-th sampled state
            observations_np : tuple(np.ndarray)
                tuple (immutable) of state observations formatted as numpy array 
                i-th element is the obervation (see observeState) at the i-th sampled state
            validation_percent : float
                percent of data to be used in validation set
            batch_size : int
                size of training batches
            num_workers : int
                number of workers to use for dataloader
        '''
        super().__init__()

        assert validation_percent >= 0 and validation_percent <= 1
        assert batch_size > 0

        assert state_samples_np.shape[0] == risk_metrics_np.shape[0] == observations_np.shape[0]
        assert len(risk_metrics_np.shape) == len(state_samples_np.shape) == len(observations_np.shape)

        self.batch_size = batch_size
        self.num_workers = num_workers

        n_data = len(state_samples_np)
        n_val = int(n_data*validation_percent)
        n_train = n_data - n_val

        # Create input and output data regularizers
        # Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(state_samples_np)
        self.observation_scaler = MinMaxScaler()
        self.observation_scaler.fit(observations_np)
        # self.output_scaler = MinMaxScaler()
        # self.output_scaler.fit(rmetrics_np)

        # scale and convert to tensor
        ssamples_scaled_pt = torch.from_numpy(self.state_scaler.transform(state_samples_np))
        lidars_scaled_pt = torch.from_numpy(self.observation_scaler.transform(observations_np))
        rmetrics_pt = torch.from_numpy(risk_metrics_np)
        
        # format into dataset
        # full_dataset = TensorDataset(ssamples_scaled_pt, rmetrics_pt)
        full_dataset = TensorDataset(lidars_scaled_pt, rmetrics_pt)

        # randomly split training and validation dataset
        self.train_dataset, self.val_dataset = random_split(full_dataset, [n_train, n_val])

        # store for later visualization use
        # self.raw_data = raw_data
        # self.raw_data_paths = dpaths

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=len(self.val_dataset), shuffle=True)


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
