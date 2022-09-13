"""
Example that visualize behavior of LRMM agent
"""

from dubins4d_reachavoid import execute_lrmm_agent
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv

env = Dubins4dReachAvoidEnv(time_accel_factor=10.0,render_mode="human")

DEFAULT_LRMM_CHKPT_FILE = (
    "/home/ross/Projects/AIIA/risk_metric_maps/" +
    "outputs/2022-09-11/22-10-44/lightning_logs/" +
    "version_0/checkpoints/epoch=2027-step=442103.ckpt"
)
DEFAULT_LRMM_ACITVE_CTRL_RISK_THRESHOLD = 0.8
DEFAULT_LRMM_DATA_PATH= (
    "/home/ross/Projects/AIIA/risk_metric_maps/" +
    "outputs/2022-09-11/21-08-47/"
)

info = execute_lrmm_agent(
    env_n_seed=(env,1),
    chkpt_file = DEFAULT_LRMM_CHKPT_FILE,
    active_ctrl_risk_threshold = DEFAULT_LRMM_ACITVE_CTRL_RISK_THRESHOLD,
    data_path = DEFAULT_LRMM_DATA_PATH,
)

print(info)