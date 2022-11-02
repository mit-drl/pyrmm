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

class ShallowRiskCBFPerceptron(nn.Module):
    def __init__(self,
        num_obs_inputs: int,
        num_state_features: int,
        num_neurons: int):
        """shallow feed-forward network with a "CBF (control barrier function) layer"

        The CBF layer outputs weights for a linear combination of state features used to compute the risk metric

        Args:
            num_obs_inputs : int
                number of inputs from observation of state, used as input at the "front" of the network
            num_state_features : int
                number of elements in the state feature vector (phi in many SVM/kernel literature),
                used as input at the "middle" of the network 
                linearly combined with the CBF-layer outputs in order to compute risk metric at model ouput
            num_neurons : int
                number of hidden units in the single hidden layer

        Ref:
            + multi-input networks in pytorch lightning: https://rosenfelder.ai/multi-input-neural-network-pytorch/
            + example of polynomial kernel (i.e. feature vector): https://en.wikipedia.org/wiki/Polynomial_kernel
        """
        super().__init__()
        self.num_state_features = num_state_features
        self.fc1 = nn.Linear(num_obs_inputs, num_neurons)
        self.fc2 = nn.Linear(num_neurons, num_state_features)

    def forward(self, observation, state_features):
        x0 = self.fc1(observation)
        x1 = torch.sigmoid(x0)
        w_cbf = self.fc2(x1)

        # w vector is now the weights on the linear combination of state_features
        rho = torch.inner(w_cbf, state_features)

        # bound risk to [0,1]
        rho = torch.sigmoid(rho)

        # return risk estimate and cbf layer weights
        return rho, w_cbf


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


class RiskCtrlMetricTrainingData():
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
    

def compile_raw_data(datapaths, verify_func:callable=None, ctrl_data=True):
    ''' extract and concat state samples, risk metrics, and observations from data files

    Assumes that data from datagen has been saved in state-risk-observation ordered format

    Args:
        datapaths : list[str]
            list of path strings to hydrazen outputs to be loaded
        verify_func : callable
            function to call to verify consistency of data
        ctrl_data : boolean
            If True, data should contain min-risk control inputs or min-risk control durations
            and all of this should be packaged together
            if False, data MAY contain min-risk control and durations,
            but it won't be packaged and returned with the states,
            risk metrics, and observations 
            (case for backward compatibility with old datasets)
    
    Returns:
        compiled_raw_data : RiskMetricTrainingData
            raw data compiled into object containing index-aligned array-like
            storage of state samples, risk_metrics, state observations, 
            min-risk ctrls, and min-risk ctrl durations
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

    if ctrl_data:
        # zip data into five immutable, indice-aligned objects
        state_samples, risk_metrics, observations, min_risk_ctrls, min_risk_ctrl_durs = tuple(zip(*concat_data))

        # create training data object with raw (non-numpy) compiled data
        compiled_raw_data = RiskCtrlMetricTrainingData(
            state_samples=state_samples,
            risk_metrics=risk_metrics,
            observations=observations,
            min_risk_ctrls=min_risk_ctrls,
            min_risk_ctrl_durs=min_risk_ctrl_durs)

    else:
        # check if dataset contains min-risk data to be ignored
        if len(concat_data[0]) == 5:
            state_samples, risk_metrics, observations, _, _ = tuple(zip(*concat_data))
        elif len(concat_data[0]) == 3:
            # for backward compatibility
            state_samples, risk_metrics, observations = tuple(zip(*concat_data))
        else: 
            raise ValueError("Unexpected data size with len: ",len(concat_data[0]))
            
        compiled_raw_data = OnlyRiskMetricTrainingData(
            state_samples=state_samples,
            risk_metrics=risk_metrics,
            observations=observations)

    return compiled_raw_data, separated_raw_data

class RiskCtrlMetricDataModule(LightningDataModule):
    def __init__(self,
        datapaths: List[str],
        val_ratio: float, 
        batch_size: int, 
        num_workers: int,
        compile_verify_func: callable=None):
        '''loads, formats, scales and checks risk-ctrl training data from torch save files
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
        self.min_risk_ctrl_scaler = MinMaxScaler()
        self.min_risk_ctrl_scaler.fit(np_data.min_risk_ctrls)
        self.min_risk_ctrl_dur_scaler = MinMaxScaler()
        self.min_risk_ctrl_dur_scaler.fit(np_data.min_risk_ctrl_durs)
        # self.output_scaler = MinMaxScaler()
        # self.output_scaler.fit(rmetrics_np)

        # scale and convert to tensor
        pt_scaled_data = RiskCtrlMetricTrainingData(
            state_samples=torch.from_numpy(self.state_scaler.transform(np_data.state_samples)),
            risk_metrics=torch.from_numpy(np_data.risk_metrics),
            observations=torch.from_numpy(self.observation_scaler.transform(np_data.observations)),
            min_risk_ctrls=torch.from_numpy(self.min_risk_ctrl_scaler.transform(np_data.min_risk_ctrls)),
            min_risk_ctrl_durs=torch.from_numpy(self.min_risk_ctrl_dur_scaler.transform(np_data.min_risk_ctrl_durs)),
        )
        
        # format scaled observations and target data (risk metrics, min-risk ctrl vars, 
        # and ctrl durations) into training dataset
        target_data = torch.cat((
            pt_scaled_data.risk_metrics, 
            pt_scaled_data.min_risk_ctrls, 
            pt_scaled_data.min_risk_ctrl_durs
            ), dim=-1
        )
        full_dataset = TensorDataset(pt_scaled_data.observations, target_data)

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
        self.control_shape = np_data.min_risk_ctrls.shape
        self.separated_raw_data = separated_raw_data

    def raw_data_to_numpy(self, raw_data: RiskCtrlMetricTrainingData):
        '''convert raw data (e.g. OMPL objects) to numpy arrays'''
        raise NotImplementedError("Must me implemented in child class")

    def verify_numpy_data(self, np_data: RiskCtrlMetricTrainingData):
        '''checks on data shape once converted to numpy form'''
        assert np_data.state_samples.shape[0] == np_data.risk_metrics.shape[0] == np_data.observations.shape[0]
        assert len(np_data.state_samples.shape) == len(np_data.risk_metrics.shape) == len(np_data.observations.shape)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=len(self.val_dataset), shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size)

class OnlyRiskMetricTrainingData():
    '''object to hold sampled states, risk metric values, and state observations
    in index-align, array-like objects

    Meant to be flexible to different data containers (lists, tuples, arrays, etc)
    used mostly like a namespace with some light error checking on sizes
    '''
    def __init__(self, 
        state_samples: ArrayLike, 
        risk_metrics: ArrayLike, 
        observations: ArrayLike):
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
        '''

        # check that amount of data in each category is equal
        assert (
            len(state_samples) == 
            len(risk_metrics) == 
            len(observations)
        )

        self.state_samples = state_samples
        self.risk_metrics = risk_metrics
        self.observations = observations

class OnlyRiskMetricDataModule(LightningDataModule):
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

    def setup(self, 
        stage: Optional[str] = None, 
        np_data: Optional[OnlyRiskMetricTrainingData]=None,
        store_raw_data: bool=True):

        if np_data is None:
            # compile raw data from pytorch files
            raw_data, separated_raw_data = compile_raw_data(
                datapaths=self.datapaths, 
                verify_func=self.compile_verify_func,
                ctrl_data=False)

            # extract useful params
            n_data = len(raw_data.state_samples)

            # convert raw data to numpy arrays
            np_data = self.raw_data_to_numpy(raw_data)

        else:
            n_data = len(np_data.state_samples)

        # verify numpy data for consistency
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
        pt_scaled_data = OnlyRiskMetricTrainingData(
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
        if store_raw_data:
            self.separated_raw_data = separated_raw_data

    def raw_data_to_numpy(self, raw_data: OnlyRiskMetricTrainingData):
        '''convert raw data (e.g. OMPL objects) to numpy arrays'''
        raise NotImplementedError("Must me implemented in child class")

    def verify_numpy_data(self, np_data: OnlyRiskMetricTrainingData):
        '''checks on data shape once converted to numpy form'''
        assert np_data.state_samples.shape[0] == np_data.risk_metrics.shape[0] == np_data.observations.shape[0]
        assert len(np_data.state_samples.shape) == len(np_data.risk_metrics.shape) == len(np_data.observations.shape)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=len(self.val_dataset), shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size)


class BaseRiskMetricModule(LightningModule):
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


class CBFLRMMModule(BaseRiskMetricModule):
    def __init__(
        self,
        num_inputs: int,
        model: nn.Module,
        optimizer: Partial[optim.Adam],
    ):
        super().__init__(num_inputs=num_inputs, model=model, optimizer=optimizer)

    def forward(self, observation, state_features):
        return self.model(observation, state_features)

    def _shared_eval_step(self, batch, batch_idx):
        raise NotImplementedError
        inputs, targets = batch
        predictions = self.model(inputs)
        loss = F.mse_loss(predictions, targets)
        mean_sqr_err = mean_squared_error(predictions, targets)
        mean_abs_err = mean_absolute_error(predictions, targets)
        max_abs_err = torch.max(torch.abs(predictions - targets))
        return loss, mean_sqr_err, mean_abs_err, max_abs_err