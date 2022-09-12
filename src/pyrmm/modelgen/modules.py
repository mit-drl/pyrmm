'''Pytorch Lightning modules for training risk metric models'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error, mean_squared_error

from pathlib import Path
from typing import List, Tuple, Optional
from numpy.typing import ArrayLike
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

def single_layer_nn_bounded_output(num_inputs: int, num_neurons: int) -> nn.Module:
    """bounds the output to the range [0,1]"""
    return nn.Sequential(
        nn.Linear(num_inputs, num_neurons),
        nn.Sigmoid(),
        nn.Linear(num_neurons, 1, bias=False),
        nn.Sigmoid(),
    )

class ShallowRiskCtrlMLP(nn.Module):
    def __init__(self,
        num_inputs: int,
        num_ctrl_dims: int, 
        num_neurons: int):
        """shallow feed-forward neural network for outputing bounded risk and control values

        Args:
            num_inputs : int
                number of inputs to neural network. Should align with observation space
            num_ctrl_dims : int
                number of control dimensions. Should align with control space
            num_neurons : int
                number of neurons in layer
        """
        super().__init__()
        self.num_ctrl_dims = num_ctrl_dims
        self.fc1 = nn.Linear(num_inputs, num_neurons)
        self.fc2 = nn.Linear(num_neurons, 2+num_ctrl_dims)

    def forward(self, x):
        # pass all inputs through first linear layer and ELU activation
        x = F.elu(self.fc1(x))
        x = self.fc2(x)

        # bound risks probabilities to range [0,1] 
        risks = torch.sigmoid(x.narrow(-1,0,1))

        # separate controls and control durations from risks
        ctrls = x.narrow(-1,1,self.num_ctrl_dims)
        durs = x.narrow(-1,-1,1)
        
        return torch.cat((risks, ctrls, durs), -1)


def linear_nn():
    return nn.Sequential(
        nn.Linear(3,1)
    )

class ResultSummary():
    '''object from holding training and testing results most salient metrics.
    Used for easy conversion to a human-readable yaml file
    '''
    def __init__(self):
        self.training = None
        self.testing = None


class RiskMetricTrainingData():
    '''object to hold sampled states, risk metric values, and state observations
    in index-align, array-like objects

    Meant to be flexible to different data containers (lists, tuples, arrays, etc)
    used mostly like a namespace with some light error checking on sizes
    '''
    def __init__(self, 
        state_samples: ArrayLike, 
        risk_metrics: ArrayLike, 
        observations: ArrayLike,
        min_risk_ctrls: ArrayLike,
        min_risk_ctrl_durs: ArrayLike):
        '''
        Args
        state_samples : ArrayLike
            state samples in their native format (i.e. OMPL State objects)
            i-th element is i-th sampled state represented as a numpy array
        risk_metrics : ArrayLike
            risk metric evaluated at corresponding sampled state
            i-th element is risk metric evaluated at the i-th sampled state
        observations : ArrayLike
            observeState outputs at each corresponding sampled state
            i-th element is the obervation (see observeState) at the i-th sampled state
        min_risk_ctrls : ArrayLike
            minimum risk control to take at a particular state as evaluated by estimateRiskMetric
            i-th element is the min-risk control at the i-th sampled state
        min_risk_ctrl_durs: ArrayLike
            duration of time to apply minimum-risk control at particular state as evaluated by estimateRiskMetrics
            i-th element is the time [sec] to apply i-th min-risk control at i-th sample state
        '''

        # check that amount of data in each category is equal
        assert (
            len(state_samples) == 
            len(risk_metrics) == 
            len(observations) ==
            len(min_risk_ctrls) ==
            len(min_risk_ctrl_durs)
        )

        self.state_samples = state_samples
        self.risk_metrics = risk_metrics
        self.observations = observations
        self.min_risk_ctrls = min_risk_ctrls
        self.min_risk_ctrl_durs = min_risk_ctrl_durs
        

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

        # need to convert zip object to list so that it can be 
        # assigned in multiple places without "emptying the iterator"
        if isinstance(cur_data, zip):
            cur_data = list(cur_data)

        separated_raw_data[str(dp)] = cur_data
        concat_data.extend(cur_data)

    # zip data into three immutable, indice-aligned objects
    state_samples, risk_metrics, observations, min_risk_ctrls, min_risk_ctrl_durs = tuple(zip(*concat_data))

    # create training data object with raw (non-numpy) compiled data
    compiled_raw_data = RiskMetricTrainingData(
        state_samples=state_samples,
        risk_metrics=risk_metrics,
        observations=observations,
        min_risk_ctrls=min_risk_ctrls,
        min_risk_ctrl_durs=min_risk_ctrl_durs)

    return compiled_raw_data, separated_raw_data


class RiskMetricDataModule(LightningDataModule):
    def __init__(self,
        datapaths: List[str],
        val_ratio: float, 
        batch_size: int, 
        num_workers: int,
        compile_verify_func: callable=None):
        '''loads, formats, scales and checks Dubins training data from torch save files
        Args:
            datapaths : List[str]
                list of paths to pytorch data files
            val_ratio : float
                ratio of data to be used in validation set. 0=no validation data, 1=all validation data
            batch_size : int
                size of training batches
            num_workers : int
                number of workers to use for dataloader
        '''
        super().__init__()
        self.datapaths = datapaths
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.compile_verify_func = compile_verify_func

        assert batch_size > 0

    def setup(self, stage: Optional[str] = None):

        # compile raw data from pytorch files
        raw_data, separated_raw_data = compile_raw_data(
            datapaths=self.datapaths, 
            verify_func=self.compile_verify_func)

        # extract useful params
        n_data = len(raw_data.state_samples)

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

        # handle training and testing separately
        if stage == "fit":
            # randomly split training and validation dataset
            assert self.val_ratio >= 0 and self.val_ratio <= 1
            n_val = int(n_data*self.val_ratio)
            n_train = n_data - n_val
            self.train_dataset, self.val_dataset = random_split(full_dataset, [n_train, n_val])

        elif stage == "test":
            self.test_dataset = full_dataset

        else:
            raise ValueError('Unexpected stage {}'.format(stage))

        # store high-level information about data in data module
        self.n_data = n_data # number of data points
        self.observation_shape = np_data.observations.shape
        self.separated_raw_data = separated_raw_data

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

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size)


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
        loss, mean_sqr_err, mean_abs_err, max_abs_err = self._shared_eval_step(batch, batch_idx)
        metrics = {'train_loss': loss, 'train_mean_sqr_err': mean_sqr_err, 'train_mean_abs_err': mean_abs_err, 'train_max_abs_err': max_abs_err}
        self.log_dict(metrics)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.print("Completed epoch {} of {}".format(self.current_epoch+1, self.trainer.max_epochs))
        self.log('avg_train_loss', avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        print('\n------------------------------\nSTARTING VALIDATION STEP\n')
        loss, mean_sqr_err, mean_abs_err, max_abs_err = self._shared_eval_step(batch, batch_idx)
        metrics = {'val_loss': loss, 'val_mean_sqr_err': mean_sqr_err, 'val_mean_abs_err': mean_abs_err, 'val_max_abs_err': max_abs_err}
        self.print("\nvalidation loss:", loss.item())
        self.log_dict(metrics)
    
    def test_step(self, batch, batch_idx):
        loss, mean_sqr_err, mean_abs_err, max_abs_err = self._shared_eval_step(batch, batch_idx)
        metrics = {'test_loss': loss, 'test_mean_sqr_err': mean_sqr_err, 'test_mean_abs_err': mean_abs_err, 'test_max_abs_err': max_abs_err}
        self.log_dict(metrics)

    def _shared_eval_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self.model(inputs)
        loss = F.mse_loss(predictions, targets)
        mean_sqr_err = mean_squared_error(predictions, targets)
        mean_abs_err = mean_absolute_error(predictions, targets)
        max_abs_err = torch.max(torch.abs(predictions - targets))
        return loss, mean_sqr_err, mean_abs_err, max_abs_err
