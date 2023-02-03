'''Pytorch Lightning modules for training risk metric models'''
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error, mean_squared_error

from pytorch_lightning import LightningModule
from hydra_zen.typing import Partial
from typing import List

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

def linear_nn():
    return nn.Sequential(
        nn.Linear(3,1)
    )

class ShallowRiskCBFPerceptron(nn.Module):
    def __init__(self,
        num_obs_inputs: int,
        num_state_features: int,
        num_neurons: int):
        """shallow feed-forward network with a "CBF (control barrier function) layer"

        This is a multi-input network: takes both observations and (local) state features, which
        are used together to compute an estimated risk value

        Technically it is also a multi-output network: outputting both the scalar risk metric 
        estimation value and the weights at the final layer of the perceptron which are then
        linear combined with the (local) state features 

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
        # to compute risk estimate
        # perform "batch dot product" with dot product on final dimension
        # see this thread for implementation: https://github.com/pytorch/pytorch/issues/18027
        # TODO: replace with vecdot in newer version of pytorch: https://pytorch.org/docs/1.13/generated/torch.linalg.vecdot.html
        rho = (w_cbf * state_features).sum(-1, keepdim=True)

        # bound risk to [0,1]
        rho = torch.sigmoid(rho)

        # return risk estimate and cbf layer weights
        return rho, w_cbf

class DeepRiskCBFPerceptron(nn.Module):
    def __init__(self,
        num_obs_inputs: int,
        num_state_features: int,
        num_neurons: List[int]):
        """Deep feed-forward network with a "CBF (control barrier function) layer"

        This is a multi-input network: takes both observations and (local) state features, which
        are used together to compute an estimated risk value

        Technically it is also a multi-output network: outputting both the scalar risk metric 
        estimation value and the weights at the final layer of the perceptron which are then
        linear combined with the (local) state features 

        The CBF layer outputs weights for a linear combination of state features used to compute the risk metric

        Args:
            num_obs_inputs : int
                number of inputs from observation of state, used as input at the "front" of the network
            num_state_features : int
                number of elements in the state feature vector (phi in many SVM/kernel literature),
                used as input at the "middle" of the network 
                linearly combined with the CBF-layer outputs in order to compute risk metric at model ouput
            num_neurons : List[int]
                number of hidden units per each layer

        Ref:
            + multi-input networks in pytorch lightning: https://rosenfelder.ai/multi-input-neural-network-pytorch/
            + example of polynomial kernel (i.e. feature vector): https://en.wikipedia.org/wiki/Polynomial_kernel
        """
        super().__init__()
        self.num_state_features = num_state_features
        self.num_neurons = num_neurons

        # define sequence of fully connected layers
        # See need for ModuleList here: https://stackoverflow.com/questions/54678896/pytorch-valueerror-optimizer-got-an-empty-parameter-list
        # self.num_layers = len(num_neurons)+1
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(num_obs_inputs, num_neurons[0]))
        self.fc_layers.append(nn.ELU())
        for i in range(1,len(self.num_neurons)):
            self.fc_layers.append(nn.Linear(num_neurons[i-1], num_neurons[i]))
            self.fc_layers.append(nn.ELU())
        self.fc_layers.append(nn.Linear(num_neurons[-1], num_state_features))

    def forward(self, observation, state_features):

        # pass observation (not state features) through all layers
        # to get CBF weights at output layer
        x = observation
        for layer in self.fc_layers:
            x = layer(x)
        w_cbf = x
        # x = self.fc_layers[0](observation)
        # x = nn.ELU()(x)
        # for i in range(1, self.num_layers-1):
        #     x = self.fc_layers[i](x)
        #     x = nn.ELU()(x)
        # w_cbf = self.fc_layers[-1](x)

        # w vector is now the weights on the linear combination of state_features
        # to compute risk estimate
        # perform "batch dot product" with dot product on final dimension
        # see this thread for implementation: https://github.com/pytorch/pytorch/issues/18027
        # TODO: replace with vecdot in newer version of pytorch: https://pytorch.org/docs/1.13/generated/torch.linalg.vecdot.html
        rho = (w_cbf * state_features).sum(-1, keepdim=True)

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

        This is a single-input (observation) and multi-output (risk estimate, min-risk control, 
        min-risk control duration) network

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

        # set example input to None to avoid cryptic pre-training errors
        self.example_input_array = None

    def forward(self, observation, state_features):

        # assumes multi-input model like ShallowRiskCBFPerceptron
        return self.model(observation, state_features)

    def _shared_eval_step(self, batch, batch_idx):

        # break batch into separate data components
        # assumes batch of StateFeatureObservationRiskDataset
        states, features, observations, risk_targets = batch

        # infer risk metric from model output
        # assumes multi-output model like ShallowRiskCBFPerceptron
        risk_estimates, cbf_weights = self.forward(observation=observations, state_features=features)

        # log control barrier layer weights mostly for validating simple 
        # example cases where weights should converge to 1.0
        self.log_dict({"mean_cbf_weights": torch.mean(cbf_weights)})
        # print("Mean CBF Weight: {}".format(torch.mean(cbf_weights,dim=0)))

        loss = F.mse_loss(risk_estimates, risk_targets)
        mean_sqr_err = mean_squared_error(risk_estimates, risk_targets)
        mean_abs_err = mean_absolute_error(risk_estimates, risk_targets)
        max_abs_err = torch.max(torch.abs(risk_estimates - risk_targets))
        return loss, mean_sqr_err, mean_abs_err, max_abs_err

