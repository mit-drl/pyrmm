"""
Class for defining an interactive guardian agent for the Dubins4d Reach-Avoid
    environment whose control policy is based on
    a learned risk metric map (LRMM) model
"""
import torch
import numpy as np

from copy import deepcopy
from numpy.typing import ArrayLike
from typing import Callable

from pyrmm.environments.dubins4d_reachavoid import \
    OS_N_RAYS_DEFAULT, Dubins4dReachAvoidEnv, \
    K_ACTIVE_CTRL, K_TURNRATE_CTRL, K_ACCEL_CTRL
from pyrmm.modelgen.modules import \
    single_layer_nn_bounded_output, OnlyRiskMetricModule, \
    ShallowRiskCtrlMLP, RiskCtrlMetricModule

class LRMMOnlyRiskDubins4dReachAvoidAgent():
    def __init__(self,
        chkpt_file: str,
        active_ctrl_risk_threshold: float,
        observation_scaler : Callable,
        u1min : float, 
        u1max : float,
        u2min : float, 
        u2max : float,
    ):
        """
        Args:
            chkpt_file : str
                absolute path to pytorch lightning checkpoint
            active_ctrl_risk_threshold : float
                risk estimate at which time active control is taken
            observation_scaler : Callable
                scikit-learn MinMax scaler used to normalize input data to model
            u1min, u1max : float
                min and max control bounds on turnrate [rad/s]
            u2min, u2max : float
                min and max control bounds on accel [m/s/s]
        """
        self.active_ctrl_risk_threshold = active_ctrl_risk_threshold
        self.observation_scaler = observation_scaler
        self.u1min = u1min
        self.u1max = u1max
        self.u2min = u2min
        self.u2max = u2max

        # form constants for reconstructing module from checkpoint
        # NOTE: this is pretty hacky, there has got to be a better
        # way to snapshot the module parameters during save-time
        self.num_inputs = OS_N_RAYS_DEFAULT + 5  # this should raise error when loading state dict if wrong
        self.num_neurons = 64 # this could be wrong if modelgen.dubins4d is modified or override is passed through hydra, but should raise error if so

        # create model object
        model = single_layer_nn_bounded_output(
            num_inputs=self.num_inputs, 
            num_neurons=self.num_neurons
        )

        # load checkpoint
        chkpt = torch.load(chkpt_file)

        # create pytorch lightning module
        # no optimizer because this is only for eval/inference
        self.lrmm = OnlyRiskMetricModule(num_inputs=self.num_inputs, model=model, optimizer=None)

        # load checkpoint into module
        self.lrmm.load_state_dict(chkpt['state_dict'])
        self.lrmm.eval()

        # specify action space from environment
        self.action_space = deepcopy(Dubins4dReachAvoidEnv.action_space)

    def get_action(self, observation: ArrayLike):
        '''given environment observation, determine risk metric
        and apply safety controller if needed

        Args:
            observation : ArrayLike
                observed state of system 
                see Dubins4dReachAvoidEnv._get_observation for array element
                definitions (e.g. time, vel, ray-casts, etc)
        '''

        obs = observation
        # create an action object for modification from sample
        action = self.action_space.sample()
        action[K_ACTIVE_CTRL] = False

        # scale observation to conform with model training scales
        obs_scaled = torch.from_numpy(
            np.float32(
                self.observation_scaler.transform(
                    obs.reshape(1,self.num_inputs)
                )
            )
        ).reshape(self.num_inputs,)

        # run inference on trained model and unpack outputs
        risk_est = self.lrmm(obs_scaled).detach().numpy()
        assert len(risk_est) == 1
        risk_est = risk_est[0]

        # determine if active control is to be taken
        if risk_est > self.active_ctrl_risk_threshold:
            action[K_ACTIVE_CTRL] = True

            if risk_est > 0.95:
                # apply full braking
                action[K_TURNRATE_CTRL] = 0.0
                action[K_ACCEL_CTRL] = self.u2min
            else:
                # apply slowing and turning control
                action[K_TURNRATE_CTRL] = (
                    self.u1max if obs[6] > obs[-1] else self.u1min
                )
                action[K_ACCEL_CTRL] = self.u2min*max(abs(obs[4]),1.0)
        
        return action

class LRMMRiskCtrlDubins4dReachAvoidAgent():
    def __init__(self,
        chkpt_file: str,
        active_ctrl_risk_threshold: float,
        observation_scaler : Callable,
        min_risk_ctrl_scaler : Callable,
        min_risk_ctrl_dur_scaler : Callable
    ):
        """
        Args:
            chkpt_file : str
                absolute path to pytorch lightning checkpoint
            active_ctrl_risk_threshold : float
                risk estimate at which time active control is taken
            data_module
        """
        self.active_ctrl_risk_threshold = active_ctrl_risk_threshold
        self.observation_scaler = observation_scaler
        self.min_risk_ctrl_scaler = min_risk_ctrl_scaler
        self.min_risk_ctrl_dur_scaler = min_risk_ctrl_dur_scaler

        # form constants for reconstructing module from checkpoint
        # NOTE: this is pretty hacky, there has got to be a better
        # way to snapshot the module parameters during save-time
        self.num_inputs = OS_N_RAYS_DEFAULT + 5  # this should raise error when loading state dict if wrong
        self.num_ctrl_dims = 2   # hacky to hardcode, but this shouldn't ever be out of alignment with data
        self.num_neurons = 64 # this could be wrong if modelgen.dubins4d is modified or override is passed through hydra, but should raise error if so

        # create model object
        model = ShallowRiskCtrlMLP(
            num_inputs=self.num_inputs, 
            num_ctrl_dims=self.num_ctrl_dims, 
            num_neurons=self.num_neurons
        )

        # load checkpoint
        chkpt = torch.load(chkpt_file)

        # create pytorch lightning module
        # no optimizer because this is only for eval/inference
        self.lrmm = RiskCtrlMetricModule(num_inputs=self.num_inputs, model=model, optimizer=None)

        # load checkpoint into module
        self.lrmm.load_state_dict(chkpt['state_dict'])
        self.lrmm.eval()

        # specify action space from environment
        self.action_space = deepcopy(Dubins4dReachAvoidEnv.action_space)

    def get_action(self, observation: ArrayLike):
        '''given environment observation, determine appropriate action from 
        inference of trained model

        Args:
            observation : ArrayLike
                observed state of system 
                see Dubins4dReachAvoidEnv._get_observation for array element
                definitions (e.g. time, vel, ray-casts, etc)
        '''

        # create an action object for modification from sample
        action = self.action_space.sample()
        action[K_ACTIVE_CTRL] = False

        # scale observation to conform with model training scales
        obs_scaled = torch.from_numpy(
            np.float32(
                self.observation_scaler.transform(
                    observation.reshape(1,self.num_inputs)
                )
            )
        ).reshape(self.num_inputs,)

        # run inference on trained model and unpack outputs
        model_out = self.lrmm(obs_scaled).detach().numpy()

        # inverse scale min-risk control and duration (risk est requires no scaling)
        risk_est = model_out[0]
        min_risk_ctrl_est = self.min_risk_ctrl_scaler.inverse_transform(
            model_out[1:1+self.num_ctrl_dims].reshape(1,self.num_ctrl_dims)
        ).reshape(self.num_ctrl_dims,)
        min_risk_ctrl_dur_est = self.min_risk_ctrl_dur_scaler.inverse_transform(
            model_out[-1].reshape(1,1)
        ).reshape(1,)[0]

        # determine if active control is to be taken
        if risk_est > self.active_ctrl_risk_threshold:
            action[K_ACTIVE_CTRL] = True
            action[K_TURNRATE_CTRL] = min_risk_ctrl_est[0]
            action[K_ACCEL_CTRL] = min_risk_ctrl_est[1]
        
        return action, min_risk_ctrl_dur_est


# if __name__ == "__main__":
#     # lrmm = LRMMDubins4dReachAvoidAgent(
#     #     pl_module_savefile="/home/ross/Projects/AIIA/risk_metric_maps/outputs/2022-09-11/22-00-45/trained_RiskCtrlMetricModule.pt")
#     lrmm = LRMMDubins4dReachAvoidAgent(
#         chkpt_file="outputs/2022-09-11/22-00-45/lightning_logs/version_0/checkpoints/epoch=7-step=1567.ckpt"
#     )
