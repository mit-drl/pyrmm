import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import cm
from pathlib import Path
from typing import List
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from hydra_zen.typing import Partial
from sklearn.preprocessing import MaxAbsScaler

import pyrmm.utils.utils as U


##############################################
################# MODEL DEF ##################
##############################################

def se2_to_numpy(se2):
    '''convert OMPL SE2StateInternal object to torch tensor'''
    return np.array([se2.getX(), se2.getY(), se2.getYaw()])

class RiskMetricDataModule(LightningDataModule):
    def __init__(self, datapaths: List, val_percent: float):
        '''loads data from torch save files
        Args:
            datapaths : list[str]
                list of path strings to hydrazen outputs to be loaded
            val_percent : float
                percent of data to be used 
        '''
        super().__init__()

        assert val_percent >= 0 and val_percent <= 1

        # convert path strings in to absolute PosixPaths
        dpaths = [Path(dp).expanduser().resolve() for dp in datapaths]

        # ensure that all data is consistent on critical configs
        RiskMetricDataModule.verify_hydrazen_rmm_data(dpaths)

        # load data objects
        data = []
        for i, dp in enumerate(dpaths):
            data.extend(torch.load(dp))
        n_data = len(data)
        n_val = int(n_data*val_percent)
        n_train = n_data - n_val

        # convert SE2StateInternal objects into numpy arrays
        ssamples, rmetrics = tuple(zip(*data))
        ssamples_np = np.concatenate([se2_to_numpy(s).reshape(1,3) for s in ssamples], axis=0)
        rmetrics_np = np.asarray(rmetrics)

        # Create input and output data regularizers
        # Ref: https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html#what-is-a-datamodule
        self.input_scaler = MaxAbsScaler()
        self.input_scaler.fit(ssamples_np)
        # self.output_scaler = MinMaxScaler()
        # self.output_scaler.fit(rmetrics_np)

        # scale and convert to tensor
        ssamples_scaled_pt = torch.from_numpy(self.input_scaler.transform(ssamples_np))
        # rmetrics_scaled_pt = torch.tensor(self.output_scaler.fit(rmetrics_np))
        rmetrics_pt = torch.tensor(rmetrics_np)
        
        # format into dataset
        full_dataset = TensorDataset(ssamples_scaled_pt, rmetrics_pt)

        # randomly split training and validation dataset
        self.train_dataset, self.val_dataset = random_split(full_dataset, [n_train, n_val])

        # RiskMetricDataModule.plot_dubins_data(data, dpaths[0])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=len(self.train_dataset))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset))

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
                ppm_file = cfg[U.SYSTEM_SETUP]['ppm_file']
                speed = cfg[U.SYSTEM_SETUP]['speed']
                turn_rad = cfg[U.SYSTEM_SETUP]['min_turn_radius']

                # record risk metric estimation critical parameters
                dur = cfg[U.DURATION]
                depth = cfg[U.TREE_DEPTH]
                policy = cfg[U.POLICY]
                brnch = cfg[U.N_BRANCHES]
            
            else:
                # check system parameters match
                assert ppm_file == cfg[U.SYSTEM_SETUP]['ppm_file']
                assert np.isclose(speed, cfg[U.SYSTEM_SETUP]['speed'])
                assert np.isclose(turn_rad, cfg[U.SYSTEM_SETUP]['min_turn_radius'])

                # check risk metric estimation critical parameters
                assert np.isclose(cfg[U.DURATION], dur)
                assert cfg[U.TREE_DEPTH] == depth
                assert cfg[U.POLICY] == policy
                assert cfg[U.N_BRANCHES] == brnch
            
    @staticmethod
    def plot_dubins_data(data, datapath, cmap='coolwarm'):

        cfg_path = datapath.parent.joinpath('.hydra','config.yaml')
        with open(cfg_path, 'r') as cfg_file:
            cfg = yaml.full_load(cfg_file)

        assert cfg[U.SYSTEM_SETUP]['ppm_file'].split('/')[-1] == 'border_640x400.ppm' 

        # unzip tuples of ssamples and rmetrics
        ssamples, rmetrics = tuple(zip(*data))

        # plot results
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        # lbl = "{}-branches, {}-depth".format(brch, dpth)
        xvals = [s.getX() for s in ssamples]
        yvals = [s.getY() for s in ssamples]
        uvals = [np.cos(s.getYaw()) for s in ssamples]
        vvals = [np.sin(s.getYaw()) for s in ssamples]
        ax.quiver(xvals, yvals, uvals, vvals, rmetrics, cmap=cmap)

        # Draw in true risk metrics and obstacles
        ax.axvline(x=0, label='Obstacle'.format(0.5), lw=4.0, c='k')
        ax.axvline(x=640, label='Obstacle'.format(0.5), lw=4.0, c='k')
        ax.axhline(y=0, label='Obstacle'.format(0.5), lw=4.0, c='k')
        ax.axhline(y=400, label='Obstacle'.format(0.5), lw=4.0, c='k')
        ax.set_title(
            "Estimated Risk Metrics for Dubins Vehicle (speed={}, turn rad={})\n".format(cfg[U.SYSTEM_SETUP]['speed'], cfg[U.SYSTEM_SETUP]['min_turn_radius']) +
            "in Constrained Box w/ uniform control sampling of duration={},\n".format(cfg[U.DURATION]) +
            "tree depth={}, and branching factor={}".format(cfg[U.TREE_DEPTH], cfg[U.N_BRANCHES]) 
        )
        ax.set_xlim([0,640])
        ax.set_ylim([0,400])
        ax.set_xlabel("x-position")
        ax.set_ylabel("y-position")
        fig.colorbar(cm.ScalarMappable(None, cmap), ax=ax, label='failure probability')
        # fig.savefig('dubins_risk_estimation', bbox_inches='tight')
        plt.show()


class RiskMetricModule(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Partial[optim.Adam]
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, inputs):
        return self.model(inputs)
    
    def configure_optimizers(self):
        return self.optimizer(self.parameters())

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = F.mse_loss(outputs, targets)
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
        self.log('validation_loss', loss)



##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################



##############################################
############### TASK FUNCTIONS ###############
##############################################

if __name__ == "__main__":

    seed_everything(0)

    # create model architecture
    n_hidden = 64
    model = nn.Sequential(
        nn.Linear(in_features=3, out_features=n_hidden),
        nn.Sigmoid(),
        nn.Linear(in_features=n_hidden, out_features=1, bias=False),
    )

    # create lightning module
    model_module = RiskMetricModule(
        model=model,
        optimizer=optim.Adam 
    )

    # create data module
    data_module = RiskMetricDataModule(
        [
            'outputs/2022-01-14/12-27-37/datagen_dubins_eb6a4_c8494.pt',
            'outputs/2022-01-14/12-29-55/datagen_dubins_eb6a4_c8494.pt',
            'outputs/2022-01-14/12-31-55/datagen_dubins_eb6a4_c8494.pt',
            'outputs/2022-01-14/12-38-06/datagen_dubins_eb6a4_c8494.pt',
            'outputs/2022-01-14/18-03-31/datagen_dubins_861c2_c8494.pt', 
        ], 
        val_percent=0.15
    )

    # create trainer
    trainer = Trainer(max_steps=512, precision=64)
    trainer.fit(model_module, data_module)
