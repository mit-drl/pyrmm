import numpy as np
import torch
from pyrmm.modelgen.modules import single_layer_nn_bounded_output, ShallowRiskCtrlMLP
from torch.utils.tensorboard import SummaryWriter

# my_net = single_layer_nn_bounded_output(4, 16)
my_net = ShallowRiskCtrlMLP(4, 2, 16)
writer = SummaryWriter()
writer.add_graph(my_net, torch.zeros(4,))
writer.close()