
"""Pytorch Lightning data modules for training risk metric models"""
import torch

from pathlib import Path
from typing import List, Optional
from numpy.typing import ArrayLike
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import MinMaxScaler


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
            
        compiled_raw_data = BaseRiskMetricTrainingData(
            state_samples=state_samples,
            risk_metrics=risk_metrics,
            observations=observations)

    return compiled_raw_data, separated_raw_data

class BaseRiskMetricTrainingData():
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
        self.n_data = len(state_samples)
        self.state_samples = state_samples
        self.risk_metrics = risk_metrics
        self.observations = observations

# class SFORData(BaseRiskMetricTrainingData):
#     """namespace-like class for indexed aligned state, features, obs, and risk data
#     SFOR = state samples(S), state feature vectors (F), observation vectors (O), risk scalars (R)
#     """
#     def __init__(self, 
#         state_samples: ArrayLike,
#         state_features: ArrayLike, 
#         risk_metrics: ArrayLike, 
#         observations: ArrayLike
#     ):

#         super().__init__(
#             state_samples=state_samples,
#             risk_metrics=risk_metrics,
#             observations=observations
#         )

#         assert len(state_features) == self.n_data
#         self.state_features = state_features

class RiskCtrlMetricTrainingData(BaseRiskMetricTrainingData):
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

        super().__init__(
            state_samples=state_samples,
            risk_metrics=risk_metrics,
            observations=observations
        )

        # check that amount of data in each category is equal
        assert (
            self.n_data == 
            len(min_risk_ctrls) ==
            len(min_risk_ctrl_durs)
        )

        self.min_risk_ctrls = min_risk_ctrls
        self.min_risk_ctrl_durs = min_risk_ctrl_durs

class StateFeatureObservationRiskDataset(Dataset):
    """Dataset that breaks data into four categories: 
        state-vectors, state-feature-vectors, observation-vectors, and scalar risk values

    The intended use is for learned control barrier functions 
    (aka learned risk metric map control barrier functions aka CBFLRMM aka RiskCBF,
    nomenclature is still being settled)
    where the observation vectors are fed as inputs at the front layer of a network
    The states---or more accurate state feature vectors, phi(x)---are fed as input to 
    an intermediate layer to compute a weighted sum (weighted by the network weights
    at that layer) which estimates the scalar risk values. 
    The risk values in the dataset are therefore the target variables

    Ref:
        for implementation reference, see https://rosenfelder.ai/multi-input-neural-network-pytorch/
    """
    def __init__(self,
        sro_data: BaseRiskMetricTrainingData,
        state_feature_map: callable
    ):
        """
        Args:
            sro_data : BaseRiskMetricTrainingData
                state-risk-observation data from index-aligned, namespace-like object
            state_feature_map : callable
                function for computing state feature vectors from state samples so that
                state features don't have to be predefined, precomuputed, and saved during data 
                generation time
        """
        self.sro_data = sro_data
        self.state_feature_map = state_feature_map

    def __len__(self):
        return self.sro_data.n_data

    def __getitem__(self, idx):
        
        # compute state feature from state sample
        state_feature = self.state_feature_map(self.sro_data.state_samples[idx])

        # return state-feature-observation-risk data 
        return (
            self.sro_data.state_samples[idx],
            state_feature,
            self.sro_data.observations[idx],
            self.sro_data.risk_metrics[idx]
        )

class BaseRiskMetricDataModule(LightningDataModule):
    def __init__(self,
        datapaths: List[str],
        val_ratio: float, 
        batch_size: int, 
        num_workers: int,
        compile_verify_func: callable=None):
        '''loads, formats, scales and checks risk training data from torch save files
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
        np_data: Optional[BaseRiskMetricTrainingData]=None,
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
            self.np_data = self.raw_data_to_numpy(raw_data)

        else:
            self.np_data = np_data
            n_data = len(np_data.state_samples)

        # verify numpy data for consistency
        self.verify_numpy_data(np_data=self.np_data)

        # scale data and format as tensors
        pt_scaled_data = self._scale_data()
        
        # format dataset of inputs and targets
        full_dataset = self.get_full_dataset(pt_scaled_data=pt_scaled_data)

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

        self._extended_setup()

    def _scale_data(self):
        """creates data regularizers and returns scaled data in tensor format
    

        Note:
            implying a private function (_) because self is modified to create scalers
        """

        # Create input and output data regularizers
        # Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(self.np_data.state_samples)
        self.observation_scaler = MinMaxScaler()
        self.observation_scaler.fit(self.np_data.observations)
        # self.output_scaler = MinMaxScaler()
        # self.output_scaler.fit(rmetrics_np)

        # scale and convert to tensor
        pt_scaled_data = BaseRiskMetricTrainingData(
            state_samples=torch.from_numpy(self.state_scaler.transform(self.np_data.state_samples)),
            risk_metrics=torch.from_numpy(self.np_data.risk_metrics),
            observations=torch.from_numpy(self.observation_scaler.transform(self.np_data.observations))
        )

        return pt_scaled_data

    def get_full_dataset(self, pt_scaled_data:BaseRiskMetricTrainingData):
        """format scaled observations and risk metrics into training dataset

        Args:
            pt_scaled_data : BaseRiskMetricTrainingData
                pytorch tensor data regularized and stored in a namespace-like object
        """
        return TensorDataset(pt_scaled_data.observations, pt_scaled_data.risk_metrics)

    def _extended_setup(self):
        """additional setup steps that may be overridden by child classes

        Note:
            implied private function because self may be modified
        """
        pass
        
    def raw_data_to_numpy(self, raw_data: BaseRiskMetricTrainingData):
        '''convert raw data (e.g. OMPL objects) to numpy arrays'''
        raise NotImplementedError("Must me implemented in child class")

    def verify_numpy_data(self, np_data: BaseRiskMetricTrainingData):
        '''checks on data shape once converted to numpy form'''
        assert np_data.state_samples.shape[0] == np_data.risk_metrics.shape[0] == np_data.observations.shape[0]
        assert len(np_data.state_samples.shape) == len(np_data.risk_metrics.shape) == len(np_data.observations.shape)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, num_workers=self.num_workers, batch_size=len(self.val_dataset), shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, batch_size=self.batch_size)


class CBFLRMMDataModule(BaseRiskMetricDataModule):
    def __init__(self,
        datapaths: List[str],
        val_ratio: float, 
        batch_size: int, 
        num_workers: int,
        state_feature_map: callable,
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
            state_feature_map : callable
                function for computing state feature vectors from state samples so that
                state features don't have to be predefined, precomuputed, and saved during data 
                generation time
        '''
        super().__init__(
            datapaths = datapaths,
            val_ratio = val_ratio,
            batch_size = batch_size,
            num_workers = num_workers,
            compile_verify_func = compile_verify_func
        )
        self.state_feature_map = state_feature_map

    def get_full_dataset(self, pt_scaled_data:BaseRiskMetricTrainingData):
        """format scaled observations and risk metrics into training dataset

        Args:
            pt_scaled_data : BaseRiskMetricTrainingData
                pytorch tensor data regularized and stored in a namespace-like object

        Refs:
            For reference, see Creating a custom PyTorch Dataset in
            https://rosenfelder.ai/multi-input-neural-network-pytorch/
        """

        return StateFeatureObservationRiskDataset(
            sro_data=pt_scaled_data,
            state_feature_map=self.state_feature_map
        )


class RiskCtrlMetricDataModule(BaseRiskMetricDataModule):
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
        super().__init__(
            datapaths = datapaths,
            val_ratio = val_ratio,
            batch_size = batch_size,
            num_workers = num_workers,
            compile_verify_func = compile_verify_func
        )

    def _scale_data(self):
        """creates data regularizers and returns scaled data in tensor format

        Note:
            implying a private function (_) because self is modified to create scalers
        """
        # Create input and output data regularizers
        # Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(self.np_data.state_samples)
        self.observation_scaler = MinMaxScaler()
        self.observation_scaler.fit(self.np_data.observations)
        self.min_risk_ctrl_scaler = MinMaxScaler()
        self.min_risk_ctrl_scaler.fit(self.np_data.min_risk_ctrls)
        self.min_risk_ctrl_dur_scaler = MinMaxScaler()
        self.min_risk_ctrl_dur_scaler.fit(self.np_data.min_risk_ctrl_durs)
        # self.output_scaler = MinMaxScaler()
        # self.output_scaler.fit(rmetrics_np)

        # scale and convert to tensor
        pt_scaled_data = RiskCtrlMetricTrainingData(
            state_samples=torch.from_numpy(self.state_scaler.transform(self.np_data.state_samples)),
            risk_metrics=torch.from_numpy(self.np_data.risk_metrics),
            observations=torch.from_numpy(self.observation_scaler.transform(self.np_data.observations)),
            min_risk_ctrls=torch.from_numpy(self.min_risk_ctrl_scaler.transform(self.np_data.min_risk_ctrls)),
            min_risk_ctrl_durs=torch.from_numpy(self.min_risk_ctrl_dur_scaler.transform(self.np_data.min_risk_ctrl_durs)),
        )

        return pt_scaled_data

    def get_full_dataset(self, pt_scaled_data:BaseRiskMetricTrainingData):
        """format scaled observations and risk metrics into training dataset

        Args:
            pt_scaled_data : BaseRiskMetricTrainingData
                pytorch tensor data regularized and stored in a namespace-like object
        """
        # format scaled observations and target data (risk metrics, min-risk ctrl vars, 
        # and ctrl durations) into training dataset
        target_data = torch.cat((
            pt_scaled_data.risk_metrics, 
            pt_scaled_data.min_risk_ctrls, 
            pt_scaled_data.min_risk_ctrl_durs
            ), dim=-1
        )
        return TensorDataset(pt_scaled_data.observations, target_data)

    def _extended_setup(self):
        """additional setup steps that may be overridden by child classes

        Note:
            implied private function because self may be modified
        """
        self.control_shape = self.np_data.min_risk_ctrls.shape