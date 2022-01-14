import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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

        # convert path strings in to absolute PosixPaths
        dpaths = [Path(dp).expanduser().resolve() for dp in datapaths]

        # ensure that all data is consistent on critical configs
        RiskMetricDataModule.verify_hydrazen_rmm_data(dpaths)

        # load data objects
        data = []
        for i, dp in enumerate(dpaths):
            data.extend(torch.load(dp))

        RiskMetricDataModule.plot_dubins_data(data, dpaths[0])

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

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################



##############################################
############### TASK FUNCTIONS ###############
##############################################

if __name__ == "__main__":
    rmm_data = RiskMetricDataModule([
        'outputs/2022-01-14/12-27-37/datagen_dubins_eb6a4_c8494.pt',
        'outputs/2022-01-14/12-29-55/datagen_dubins_eb6a4_c8494.pt',
        'outputs/2022-01-14/12-31-55/datagen_dubins_eb6a4_c8494.pt'
        # 'outputs/2022-01-14/11-25-51/datagen_dubins_f8951_ebfd4.pt',
        # 'outputs/2022-01-14/11-46-12/datagen_dubins_e6ecd_ebfd4.pt',
        # 'outputs/2022-01-14/11-53-42/datagen_dubins_e6ecd_ebfd4.pt',
        # 'outputs/2022-01-14/10-36-03/datagen_dubins_739e6_ffce1.pt',
        # 'outputs/2022-01-14/10-42-04/datagen_dubins_739e6_ffce1.pt',
        # 'outputs/2022-01-14/10-47-03/datagen_dubins_739e6_ffce1.pt',
        # './outputs/2022-01-13/15-23-10',
        # './outputs/2022-01-14/10-02-24',
        # './outputs/2022-01-14/10-09-51'
    ])