"""
Example that visualize behavior of LRMM agent
"""

from dubins4d_reachavoid import execute_lrmm_agent
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv, \
    CS_DTHETAMIN, CS_DTHETAMAX, CS_DVMIN, CS_DVMAX

env = Dubins4dReachAvoidEnv(time_accel_factor=10.0,render_mode="human")

DEFAULT_LRMM_CHKPT_FILE = (
    "/home/ross/Projects/AIIA/risk_metric_maps/" +
    "outputs/2022-09-13/18-04-23/lightning_logs/" +
    "version_0/checkpoints/epoch=2027-step=20279.ckpt"
)
DEFAULT_LRMM_ACITVE_CTRL_RISK_THRESHOLD = 0.75
DEFAULT_LRMM_DATA_PATH= (
    "/home/ross/Projects/AIIA/risk_metric_maps/" +
    "outputs/2022-09-13/17-35-48/"
)
info = execute_lrmm_agent((env,14),
    chkpt_file = DEFAULT_LRMM_CHKPT_FILE,
    active_ctrl_risk_threshold = DEFAULT_LRMM_ACITVE_CTRL_RISK_THRESHOLD,
    data_path = DEFAULT_LRMM_DATA_PATH,
    u1min = CS_DTHETAMIN,
    u1max = CS_DTHETAMAX,
    u2min = CS_DVMIN,
    u2max = CS_DVMAX)

print(info)