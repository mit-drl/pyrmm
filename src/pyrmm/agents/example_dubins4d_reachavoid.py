import numpy as np
from pyrmm.agents.dubins4d_reachavoid_agent import LRMMDubins4dReachAvoidAgent

def run_example():
    chkpt_file = (
        "/home/ross/Projects/AIIA/risk_metric_maps/" +
        "outputs/2022-09-11/22-10-44/lightning_logs/" +
        "version_0/checkpoints/epoch=2027-step=442103.ckpt"
    )
    active_ctrl_thresh = 0.5

    lrmm_agent = LRMMDubins4dReachAvoidAgent(
        chkpt_file=chkpt_file, 
        active_ctrl_risk_threshold=active_ctrl_thresh)

    # generate an observation of a nearby obstacle
    obs = np.zeros(17, dtype=np.float32)
    obs[1] = 10.0   # x relative to goal
    obs[4] = 2.0    # speed
    obs[5] = 0.001  # forward ray-cast
    obs[6] = 0.01   # first left ray-cast
    obs[-1] = 0.01  # first right ray-cast

    action, ctrl_dur = lrmm_agent.get_action(obs)

    print(action, ctrl_dur)


if __name__ == "__main__":
    run_example()