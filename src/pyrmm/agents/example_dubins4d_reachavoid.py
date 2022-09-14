import numpy as np
import pyrmm.utils.utils as U
from pyrmm.modelgen.dubins4d import Dubins4dReachAvoidDataModule
from pyrmm.agents.dubins4d_reachavoid_agent import LRMMDubins4dReachAvoidAgent

def run_example():

    datapaths = U.get_abs_pt_data_paths("outputs/2022-09-11/21-08-47/")
    data_module = Dubins4dReachAvoidDataModule(datapaths,0,1,0,None)
    data_module.setup('test')

    chkpt_file = (
        "/home/ross/Projects/AIIA/risk_metric_maps/" +
        "outputs/2022-09-11/22-10-44/lightning_logs/" +
        "version_0/checkpoints/epoch=2027-step=442103.ckpt"
    )
    active_ctrl_thresh = 0.5

    lrmm_agent = LRMMDubins4dReachAvoidAgent(
        chkpt_file=chkpt_file, 
        active_ctrl_risk_threshold=active_ctrl_thresh,
        observation_scaler=data_module.observation_scaler,
        min_risk_ctrl_scaler=data_module.min_risk_ctrl_scaler,
        min_risk_ctrl_dur_scaler=data_module.min_risk_ctrl_dur_scaler)

    # generate an observation of a nearby obstacle
    obs = np.zeros(17, dtype=np.float32)
    obs[1] = 10.0   # x relative to goal
    obs[4] = 2.0    # speed
    obs[5:] = 5*np.ones(12,)
    obs[7] = 0.1 # 90-deg ray
    obs[8] = 0.1 # 90-deg ray
    # obs[5] = 0.001  # forward ray-cast
    # obs[6] = 0.01   # first left ray-cast
    # obs[-1] = 0.01  # first right ray-cast

    action, ctrl_dur = lrmm_agent.get_action(obs)

    print(action, ctrl_dur)


if __name__ == "__main__":
    run_example()