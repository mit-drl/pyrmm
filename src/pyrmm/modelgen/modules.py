'''Pytorch Lightning modules for training risk metric models'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from typing import List, Tuple
from types import SimpleNamespace
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, LightningModule
from hydra_zen.typing import Partial
from sklearn.preprocessing import MinMaxScaler

import pyrmm.utils.utils as U

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

class RiskMetricTrainingData():
    '''object to hold sampled states, risk metric values, and state observations
    in index-align, array-like objects

    Meant to be flexible to different data containers (lists, tuples, arrays, etc)
    used mostly like a namespace with some light error checking on sizes
    '''
    def __init__(self, state_samples, risk_metrics, observations):
        '''
        Args
        state_samples : array-like
            state samples in their native format (i.e. OMPL State objects)
            i-th element is i-th sampled state represented as a numpy array
        risk_metrics : array-like
            risk metric evaluated at corresponding sampled state
            i-th element is risk metric evaluated at the i-th sampled state
        observations : array-like
            observeState outputs at each corresponding sampled state
            i-th element is the obervation (see observeState) at the i-th sampled state
        '''

        # check that amount of data in each category is equal
        assert len(state_samples) == len(risk_metrics) == len(observations)

        self.state_samples = state_samples
        self.risk_metrics = risk_metrics
        self.observations = observations
        

def compile_raw_data(datapaths, verify_func:callable=None):
    ''' extract and concat state samples, risk metrics, and observations from data files

    Assumes that data from datagen has been saved in state-risk-observation ordered format

    Args:
        datapaths : list[str]
            list of path strings to hydrazen outputs to be loaded
        verify_func : callable
                function to call to verify consistency of data
    
    Returns:
        compiled_raw_data : RiskMetricTrainingData
            raw data compiled into object containing index-aligned array-like
            storage of state samples, risk_metrics, and state observations
        separated_raw_data : dict
            raw data separated by data files, useful for visualization
    '''
    # convert path strings in to absolute PosixPaths
    dpaths = [Path(dp).expanduser().resolve() for dp in datapaths]

    # ensure that all data is consistent on critical configs
    if verify_func is not None:
        verify_func(dpaths)

    # load data objects
    separated_raw_data = dict()
    concat_data = []
    for dp in dpaths:
        cur_data = torch.load(dp)
        separated_raw_data[str(dp)] = cur_data
        concat_data.extend(cur_data)

    # zip data into three immutable, indice-aligned objects
    state_samples, risk_metrics, observations = tuple(zip(*concat_data))

    # create training data object with raw (non-numpy) compiled data
    compiled_raw_data = RiskMetricTrainingData(
        state_samples=state_samples,
        risk_metrics=risk_metrics,
        observations=observations)

    return compiled_raw_data, separated_raw_data


class RiskMetricDataModule(LightningDataModule):
    def __init__(self,
        datapaths: List[str],
        val_percent: float, 
        batch_size: int, 
        num_workers: int,
        compile_verify_func: callable=None):
        '''loads, formats, scales and checks Dubins training data from torch save files
        Args:
            datapaths : List[str]
                list of paths to pytorch data files
            val_percent : float
                percent of data to be used in validation set
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

        # compile raw data from pytorch files
        raw_data, separated_raw_data = compile_raw_data(datapaths=datapaths, verify_func=compile_verify_func)

        # extract useful params
        n_data = len(raw_data.state_samples)
        n_val = int(n_data*val_percent)
        n_train = n_data - n_val

        # convert raw data to numpy arrays
        np_data = self.raw_data_to_numpy(raw_data)
        self.verify_numpy_data(np_data=np_data)

        # Create input and output data regularizers
        # Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(np_data.state_samples)
        self.observation_scaler = MinMaxScaler()
        self.observation_scaler.fit(np_data.observations)
        # self.output_scaler = MinMaxScaler()
        # self.output_scaler.fit(rmetrics_np)

        # scale and convert to tensor
        pt_scaled_data = RiskMetricTrainingData(
            state_samples=torch.from_numpy(self.state_scaler.transform(np_data.state_samples)),
            risk_metrics=torch.from_numpy(np_data.risk_metrics),
            observations=torch.from_numpy(self.observation_scaler.transform(np_data.observations))
        )
        
        # format scaled observations and risk metrics into training dataset
        full_dataset = TensorDataset(pt_scaled_data.observations, pt_scaled_data.risk_metrics)

        # randomly split training and validation dataset
        self.train_dataset, self.val_dataset = random_split(full_dataset, [n_train, n_val])

        # store high-level information about data in data module
        # ret = SimpleNamespace()
        self.n_data = n_data # number of data points
        self.observation_shape = np_data.observations.shape
        self.separated_raw_data = separated_raw_data
        # return ret

    def raw_data_to_numpy(self, raw_data: RiskMetricTrainingData):
        '''convert raw data (e.g. OMPL objects) to numpy arrays'''
        raise NotImplementedError("Must me implemented in child class")

    def verify_numpy_data(self, np_data: RiskMetricTrainingData):
        '''checks on data shape once converted to numpy form'''
        assert np_data.state_samples.shape[0] == np_data.risk_metrics.shape[0] == np_data.observations.shape[0]
        assert len(np_data.state_samples.shape) == len(np_data.risk_metrics.shape) == len(np_data.observations.shape)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=len(self.val_dataset), shuffle=True)


class RiskMetricModule(LightningModule):
    def __init__(
        self,
        num_inputs: int,
        model: nn.Module,
        optimizer: Partial[optim.Adam],
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.example_input_array = torch.rand(32,num_inputs,dtype=torch.double)

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
