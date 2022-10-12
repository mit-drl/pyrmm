import numpy as np
from pathlib import Path
import pyrmm.utils.utils as U
from pyrmm.modelgen.modules import compile_raw_data

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

tdp = 'outputs/2022-07-28/11-22-06/'
test_datapaths = U.get_abs_pt_data_paths(tdp)

_, separated_raw_test_data = compile_raw_data(test_datapaths, None, False)

test_dp = np.random.choice(list(separated_raw_test_data.keys()))

U.plot_dubins_data(Path(test_dp), desc="", data=separated_raw_test_data[test_dp], show=True)

