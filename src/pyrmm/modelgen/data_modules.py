
"""Pytorch Lightning data modules for training risk metric models"""
import torch
import numpy as np

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
            If True, data should contain min-risk control inputs and min-risk control durations
            and all of this should be packaged together
            if False, data MAY contain min-risk control and durations,
            but it won't be packaged and returned with the states,
            risk metrics, and observations 
            (case for backward compatibility with old datasets)
    
    Returns:
        compiled_raw_data : BaseRiskMetricTrainingData
            raw data compiled into object containing index-aligned array-like
            storage of state samples, risk_metrics, state observations, 
            min-risk ctrls, and min-risk ctrl durations
        separated_raw_data : dict
            raw data separated by data files
            useful for visualization and post-hoc data generation
            (e.g. creating new training data from data-pairs)
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

class SFORData(BaseRiskMetricTrainingData):
    """namespace-like class for indexed aligned state, features, obs, and risk data
    SFOR = state samples(S), state feature vectors (F), observation vectors (O), risk scalars (R)
    """
    def __init__(self, 
        state_samples: ArrayLike,
        state_features: ArrayLike, 
        risk_metrics: ArrayLike, 
        observations: ArrayLike
    ):

        super().__init__(
            state_samples=state_samples,
            risk_metrics=risk_metrics,
            observations=observations
        )

        assert len(state_features) == self.n_data
        self.state_features = state_features

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
    def __init__(self, sfor_data: SFORData):
        """
        Args:
            sfor_data : SFORData
                state-feature-observation-risk data from index-aligned, namespace-like object
        """
        self.sfor_data = sfor_data

    def __len__(self):
        return self.sfor_data.n_data

    def __getitem__(self, idx):

        # return state-feature-observation-risk data 
        return (
            self.sfor_data.state_samples[idx],
            self.sfor_data.state_features[idx],
            self.sfor_data.observations[idx],
            self.sfor_data.risk_metrics[idx]
        )

class deprecated_StateFeatureObservationRiskDataset(Dataset):
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

    Deprecated in favor of implementation that does not evaluate state features during
    data point access in __getitem__

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

# class LocalStateFeatureObservationRiskDataset(Dataset):
#     """Dataset that breaks LSFOR data into four categories: 
#         local-state-vectors (LS)
#         local-state-feature-vectors (F)
#         observation-vectors (O)
#         and scalar risk values (R)

#     This dataset is similar to StateFeatureObservationRiskDataset except that the state vector
#     and corresponding feature vector are represented in a local reference frame relative to a 
#     reference state. 

#     This means that each absolute state in the "raw" dataset corresponds to N data points in the
#     dataset because each state can be evaluated in N different local frames.

#     This form of dataset is intended for use in risk-metric control barrier function models where
#     risk metrics are modeled as localized, continuously differentiable functions of the observation 
#     at a particular state (input to first layer a neural network) and the feature vector of states
#     in the local coordinate system of the reference state.

#     Ref:
#         for implementation reference, see https://rosenfelder.ai/multi-input-neural-network-pytorch/
#     """
#     def __init__(self,
#         sro_data: BaseRiskMetricTrainingData,
#         state_feature_map: callable,
#         local_coord_map: callable
#     ):
#         """
#         Args:
#             sro_data : BaseRiskMetricTrainingData
#                 state-risk-observation data from index-aligned, namespace-like object
#             state_feature_map : callable
#                 function for computing state feature vectors from state samples so that
#                 state features don't have to be predefined, precomuputed, and saved during data 
#                 generation time
#             local_coord_map : callable
#                 function for mapping an absolute state to a local coordinate frame.
#                 For example this may be a simple subtraction of the reference state,
#                 but it may require more sophisticated functions for non-Euclidean 
#                 spaces
#         """
#         self.sro_data = sro_data
#         self.state_feature_map = state_feature_map
#         self.local_coord_map = local_coord_map

#     def __len__(self):
#         """each absoluted state corresponds to N data points becasue
#         each state can be mapped to the local reference frame of every other state.
#         Therefore N*2 total data points
#         """
#         return self.sro_data.n_data * self.sro_data.n_data

#     def __getitem__(self, idx):
#         """dataset is indexed such that an unraveled 2D index is in 
#         (abs_state, ref_state) order
#         """

#         # unravel index to identify the absolute state and reference state
#         n_raw_data = self.sro_data.n_data
#         abs_idx, ref_idx = np.unravel_index(idx, (n_raw_data, n_raw_data))

#         # compute local state relative to reference state
#         local_state = self.local_coord_map(
#             abs_state=self.sro_data.state_samples[abs_idx],
#             ref_state=self.sro_data.state_samples[ref_idx])
        
#         # compute state feature from state sample
#         local_state_feature = self.state_feature_map(local_state)

#         # return state-feature-observation-risk data 
#         return (
#             local_state,
#             local_state_feature,
#             self.sro_data.observations[abs_idx],
#             self.sro_data.risk_metrics[abs_idx]
#         )

class BaseRiskMetricDataModule(LightningDataModule):
    def __init__(self,
        datapaths: List[str],
        val_ratio: float, 
        batch_size: int, 
        num_workers: int,
        compile_verify_func: callable=None):
        '''prelim member variable instantiation, most init processes occur in setup()
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
        '''loads, formats, scales and checks risk training data from torch save files

        These steps are not performed at __init__() so that training and testing
        can be handled separately but with the same DataModule instance 

        Args:
            stage : Optional[str]
                define if the setup if for training or testing purposes
            np_data : Optional[BaseRiskMetricTrainingData]
                "raw" data that has already been formatted into numpy objects
                using a namespace-like-object BaseRiskMetricTrainingData
                if None is provided, then raw data compiled from datapaths 
            store_raw_data : bool
                If True, the raw data stored in a instance variable dictionary
                separated by the distinct file datapaths from which the data came
                If False, no such instance variable is set, which saves memory
        '''

        if np_data is None:
            # compile raw data from pytorch files
            raw_data, separated_raw_data = compile_raw_data(
                datapaths=self.datapaths, 
                verify_func=self.compile_verify_func,
                ctrl_data=False)

            # convert raw data to numpy arrays
            np_data = self.raw_data_to_numpy(raw_data)

        else:
            pass

        # verify numpy data for consistency
        self.verify_numpy_data(np_data=np_data)

        # scale data and format as tensors
        pt_scaled_data = self._scale_data()
        
        # format dataset of inputs and targets
        full_dataset = self.format_scaled_dataset(pt_scaled_data=pt_scaled_data)
        n_data = len(full_dataset)

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

        # self._extended_setup()

    def _scale_data(self, np_data:BaseRiskMetricTrainingData)->BaseRiskMetricTrainingData:
        """creates data regularizers and returns scaled data in tensor format

        Args:
            np_data : BaseRiskMetricTrainingData
                unscaled data in numpy arrays stored in namespace-like object
            
        Returns
            pt_scaled_data : BaseRiskMetricTrainingData
                scaled data in pytorch tensors stored in namespace-like object
    
        Notes:
            + Implying a private function (_) because self is modified to create scalers.
            + The risk metrics (i.e. "outputs") are not scaled because they are assumed to already
                be in range 0-1. This is done so that we aren't keeping track of a superfluous
                data scaler, but perhaps it is better to have it just for consistency...?
            
        """

        # Create input and output data regularizers
        # Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(np_data.state_samples)
        self.observation_scaler = MinMaxScaler()
        self.observation_scaler.fit(np_data.observations)
        # self.output_scaler = MinMaxScaler()
        # self.output_scaler.fit(rmetrics_np)

        # scale and convert to tensor
        pt_scaled_data = BaseRiskMetricTrainingData(
            state_samples=torch.from_numpy(self.state_scaler.transform(np_data.state_samples)),
            risk_metrics=torch.from_numpy(np_data.risk_metrics),
            observations=torch.from_numpy(self.observation_scaler.transform(np_data.observations))
        )

        return pt_scaled_data

    def format_scaled_dataset(self, pt_scaled_data:BaseRiskMetricTrainingData):
        """format scaled observations and risk metrics into training dataset

        Args:
            pt_scaled_data : BaseRiskMetricTrainingData
                pytorch tensor data regularized and stored in a namespace-like object
        """
        return TensorDataset(pt_scaled_data.observations, pt_scaled_data.risk_metrics)

    # def _extended_setup(self):
    #     """additional setup steps that may be overridden by child classes

    #     Note:
    #         implied private function because self may be modified
    #     """
    #     pass
        
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

class LSFORDataModule(BaseRiskMetricDataModule):
    def __init__(self,
        datapaths: List[str],
        val_ratio: float, 
        batch_size: int, 
        num_workers: int,
        state_feature_map: callable,
        local_states_datagen: callable,
        compile_verify_func: callable=None):
        '''Data module for local-state (LS), local-feature-vector (F), observation (O), and risk metric (R) datasets

        Loads, formats, scales, error-checks, defines local coordinate transform, and defines state feature map of 
        state-risk-observation training data from torch save files

        In spite of the naming convention/acronym, this class is very closely related to the CBFLRMMDataModule
        except that it also defines the local coordinate transformation

        Intended for use within a control barrier function (CBF) with learned risk metric map (LRMM) network model
        
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
            local_states_datagen : callable
                function for expanding data by generating relative states expressed in local reference frames
                Expansions creates absolute-state and reference-state pairs based on some filtering critera (e.g. distance)
                and then maps an absolute state to a local coordinate frame of the reference state.
                For example this may be a simple subtraction of the reference state,
                but it may require more sophisticated functions for non-Euclidean 
                spaces
        '''
        super().__init__(
            datapaths = datapaths,
            val_ratio = val_ratio,
            batch_size = batch_size,
            num_workers = num_workers,
            compile_verify_func = compile_verify_func
        )
        self.state_feature_map = state_feature_map
        self.local_states_datagen = local_states_datagen

    def setup(self, stage: Optional[str] = None):
        """ Load data and then expand the dataset using state-pairing
        """

        # extract raw data separated by datafile (i.e. datagen
        # environment instance)
        _, separated_raw_data = compile_raw_data(
            datapaths=self.datapaths, 
            verify_func=self.compile_verify_func,
            ctrl_data=False)

        # using the separated raw data from the first setup pass
        # generate new data points based upon the localization process
        # that uses state-pairs to describe relative states in local
        # reference frames of other states
        np_local_states_data = self.local_states_datagen(separated_raw_data=separated_raw_data)
        n_data = len(np_local_states_data.state_samples)

        # compute state features on localized states
        state_features = n_data*[None]
        for i in range(n_data):
            state_features[i] = self.state_feature_map(np_local_states_data.state_samples[i])
        state_features = np.asarray(state_features)

        np_sfor_data = SFORData(
            state_samples=np_local_states_data.state_samples,
            state_features=state_features,
            observations=np_local_states_data.observations,
            risk_metrics=np_local_states_data.risk_metrics)

        # ensure consistency in shape of unscaled numpy data
        self.verify_numpy_data(np_data=np_sfor_data)

        # scale data for better numerical performance and convert to tensors
        pt_scaled_sfor_data = self._scale_data(np_data=np_sfor_data)

        # format dataset of inputs and targets
        full_dataset = self.format_scaled_dataset(pt_scaled_data=pt_scaled_sfor_data)
        assert len(full_dataset) == n_data

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
        self.observation_shape = np_sfor_data.observations.shape

    def verify_numpy_data(self, np_data: SFORData):
        '''checks on data shape once converted to numpy form'''
        assert np_data.state_samples.shape[0] == np_data.state_features.shape[0] == \
             np_data.risk_metrics.shape[0] == np_data.observations.shape[0]
        assert len(np_data.state_samples.shape) == len(np_data.state_features.shape) == \
            len(np_data.risk_metrics.shape) == len(np_data.observations.shape)

    def _scale_data(self, np_data:SFORData)->SFORData:
        """creates data regularizers and returns scaled data in tensor format

        Args:
            np_data : SFORData
                unscaled data in numpy arrays stored in namespace-like object
            
        Returns
            pt_scaled_data : SFORData
                scaled data in pytorch tensors stored in namespace-like object
    
        Notes:
            + Implying a private function (_) because self is modified to create scalers.
            + The risk metrics (i.e. "outputs") are not scaled because they are assumed to already
                be in range 0-1. This is done so that we aren't keeping track of a superfluous
                data scaler, but perhaps it is better to have it just for consistency...?
            
        """

        # Create state, feature (input), observation (input) data regularizers,
        # Do not create risk (output) regulizer 
        # Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule
        self.state_scaler = MinMaxScaler()
        self.state_scaler.fit(np_data.state_samples)
        # self.feature_scaler = MinMaxScaler()
        # self.feature_scaler.fit(np_data.state_features)
        self.observation_scaler = MinMaxScaler()
        self.observation_scaler.fit(np_data.observations)
        # self.risk_metric_scaler = MinMaxScaler()
        # self.risk_metric_scaler.fit(rmetrics_np)

        # scale and convert to tensor
        pt_scaled_data = SFORData(
            state_samples=torch.from_numpy(self.state_scaler.transform(np_data.state_samples)),
            # state_features=torch.from_numpy(self.feature_scaler.transform(np_data.state_features)),
            state_features=torch.from_numpy(np_data.state_features),
            observations=torch.from_numpy(self.observation_scaler.transform(np_data.observations)),
            risk_metrics=torch.from_numpy(np_data.risk_metrics),
        )

        return pt_scaled_data

    def format_scaled_dataset(self, pt_scaled_data:SFORData):
        """format scaled observations and risk metrics into training dataset

        Args:
            pt_scaled_data : SFOR
                pytorch tensor data regularized and stored in a namespace-like object
        """
        return StateFeatureObservationRiskDataset(sfor_data=pt_scaled_data)


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

    def format_scaled_dataset(self, pt_scaled_data:BaseRiskMetricTrainingData):
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

class deprecated_CBFLRMMDataModule(BaseRiskMetricDataModule):
    def __init__(self,
        datapaths: List[str],
        val_ratio: float, 
        batch_size: int, 
        num_workers: int,
        state_feature_map: callable,
        compile_verify_func: callable=None):
        '''loads, formats, scales, checks, and applies feature map of state-risk-observation training data from torch save files

        Intended for use within a control barrier function (CBF) with learned risk metric map (LRMM) network model

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

    def format_scaled_dataset(self, pt_scaled_data:BaseRiskMetricTrainingData):
        """format scaled observations and risk metrics into training dataset

        Args:
            pt_scaled_data : BaseRiskMetricTrainingData
                pytorch tensor data regularized and stored in a namespace-like object

        Refs:
            For reference, see Creating a custom PyTorch Dataset in
            https://rosenfelder.ai/multi-input-neural-network-pytorch/
        """

        return deprecated_StateFeatureObservationRiskDataset(
            sro_data=pt_scaled_data,
            state_feature_map=self.state_feature_map
        )