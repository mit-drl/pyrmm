import torch
import hydra
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from hydra_zen import builds, make_custom_builds_fn, make_config, instantiate
from hydra_zen.typing import Partial
from sklearn.preprocessing import MinMaxScaler

_CONFIG_NAME = "simple_regression"

# Generate data
_X_MAX = 640
_N_DATA = 10000
_N_TRAIN = 9000
_X_DATA = _X_MAX*np.random.rand(_N_DATA).reshape(-1,1)
# _Y_DATA = (_X_DATA/_X_MAX).reshape(_N_DATA,)
_Y_DATA = (_X_DATA/_X_MAX)  # DON'T RESHAPE!! Make the same shape as x


class SimpleRegressionDataModule(LightningDataModule):
    def __init__(self, batch_size: int):
        '''loads regression data
        '''
        super().__init__()

        assert batch_size > 0

        self.batch_size = batch_size

        # Create input and output data regularizers
        # Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule
        self.input_scaler = MinMaxScaler()
        self.input_scaler.fit(_X_DATA)

        # scale and convert to tensor
        x_data_scaled_pt = torch.from_numpy(self.input_scaler.transform(_X_DATA))
        y_data_pt = torch.tensor(_Y_DATA)

        # randomly split training and validation dataset
        self.train_dataset = TensorDataset(x_data_scaled_pt[:_N_TRAIN], y_data_pt[:_N_TRAIN])
        self.val_dataset = TensorDataset(x_data_scaled_pt[-(_N_DATA-_N_TRAIN):], y_data_pt[-(_N_DATA-_N_TRAIN):])

    def train_dataloader(self):
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=True)
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

class SimpleRegressionModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Partial[optim.Adam],
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.example_input_array = torch.rand(32,1,dtype=torch.double)

    def forward(self, inputs):
        return self.model(inputs)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = F.mse_loss(outputs, targets)
        # self.print("\ntraining loss:", loss.item())
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        print('\n------------------------------\nSTARTING VALIDATION STEP\n')
        inputs, targets = batch
        pred = self.model(inputs)
        loss = F.mse_loss(pred, targets)
        self.print("\nvalidation loss:", loss.item())
        self.print("Model weights: {}, bias: {}".format(self.model[0].weight, self.model[0].bias))
        self.log('validation_loss', loss)

def linear_nn():
    return nn.Sequential(
        nn.Linear(1,1, bias=False)
    )

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

DataConf = builds(SimpleRegressionDataModule, batch_size=64)

ModelConf = builds(linear_nn)

OptimConf = pbuilds(optim.Adam)

PLModuleConf = builds(SimpleRegressionModule, model=ModelConf, optimizer=OptimConf)

TrainerConf = pbuilds(Trainer, 
    max_epochs=32, 
    precision=64, 
    reload_dataloaders_every_epoch=True, 
    progress_bar_refresh_rate=0)

ExperimentConfig = make_config(
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

    # instantiate the experiment objects
    obj = instantiate(cfg)

    # train the model
    trainer = obj.trainer()
    trainer.fit(obj.pl_module, obj.data_module)

    # randomly sample test data for visualization
    # convert SE2StateInternal objects into numpy arrays
    obj.pl_module.eval()
    print('\n\nTEST EVALS\n\n')
    test_inpt_np = np.array([
        [0], 
        [320], 
        [640], 
        ])
    t0_scaled_pt = torch.from_numpy(obj.data_module.input_scaler.transform(test_inpt_np))
    t0_pred_pt = obj.pl_module(t0_scaled_pt)
    for i, t in enumerate(test_inpt_np):
        print("orig state: {}\nscaled state: {}\nrisk pred: {}\n=================\n".format(t, t0_scaled_pt.numpy()[i], t0_pred_pt.detach().numpy()[i]))


##############################################
############### TASK FUNCTIONS ###############
##############################################

if __name__ == "__main__":
    task_function()
