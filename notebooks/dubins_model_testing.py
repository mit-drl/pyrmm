import torch
from pyrmm.modelgen.modules import RiskMetricModule, single_layer_nn
from pyrmm.modelgen.dubins import DubinsPPMDataModule

# create datamodule from test dataset
data_module = DubinsPPMDataModule(['outputs/2022-07-28/11-22-06/datagen_dubins_c2b43_3d307_maze_0_type_4.pt'],None,32,4,None)
data_module.setup(stage='test')

# create pytorch lightning module from pre-trained model
num_inputs = data_module.observation_shape[1]
pl_model = single_layer_nn(num_inputs=num_inputs, num_neurons=64)
pl_module = RiskMetricModule.load_from_checkpoint(
    checkpoint_path='outputs/2022-03-18/10-58-27/lightning_logs/version_0/checkpoints/epoch=8191-step=4456447.ckpt',
    num_inputs=num_inputs, model=pl_model, optimizer=None)

# evaluate pre-trained model 
pl_module.eval()
pred_risk_metrics = pl_module(data_module.test_dataset.tensors[0].float())
print('predicted data range: {} - {}'.format(torch.min(pred_risk_metrics), torch.max(pred_risk_metrics)))
print('max prediction error: {}'.format(torch.max(torch.abs(pred_risk_metrics - data_module.test_dataset.tensors[1]))))
