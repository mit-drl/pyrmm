import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List
from pytorch_lightning import LightningDataModule

import pyrmm.utils.utils as U


##############################################
################# MODEL DEF ##################
##############################################

class RiskMetricDataModule(LightningDataModule):
    def __init__(self, datapaths: List):
        '''loads data from torch save files
        Args:
            datapaths : list[str]
                list of path strings to hydrazen outputs to be loaded
        '''
        super().__init__()

        # convert path strings in to absolute PosixPaths'./outputs/2022-01-13/15-23-10',
        # './outputs/2022-01-14/10-02-24',
        # './outputs/2022-01-14/10-09-51'
        dpaths = [Path(dp).expanduser().resolve() for dp in datapaths]

        # ensure that all data is consistent on critical configs
        RiskMetricDataModule.verify_hydrazen_rmm_data(dpaths)

        # load data objects
        dobjs = len(dpaths) * [None]
        for i, dp in enumerate(dpaths):
            dobjs[i] = torch.load(dp)
            print(dobjs[i])

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
            
            else:
                # check system parameters match
                assert ppm_file == cfg[U.SYSTEM_SETUP]['ppm_file']
                assert np.isclose(speed, cfg[U.SYSTEM_SETUP]['speed'])
                assert np.isclose(turn_rad, cfg[U.SYSTEM_SETUP]['min_turn_radius'])

                # check risk metric estimation critical parameters
                assert np.isclose(cfg[U.DURATION], dur)
                assert cfg[U.TREE_DEPTH] == depth
                assert cfg[U.POLICY] == policy
            
            

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################



##############################################
############### TASK FUNCTIONS ###############
##############################################

if __name__ == "__main__":
    rmm_data = RiskMetricDataModule([
        'outputs/2022-01-14/10-36-03/datagen_dubins_739e6_ffce1.pt',
        'outputs/2022-01-14/10-42-04/datagen_dubins_739e6_ffce1.pt',
        'outputs/2022-01-14/10-47-03/datagen_dubins_739e6_ffce1.pt',
        # './outputs/2022-01-13/15-23-10',
        # './outputs/2022-01-14/10-02-24',
        # './outputs/2022-01-14/10-09-51'
    ])