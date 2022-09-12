import torch
import torch.nn as nn
import torch.nn.functional as F
from pyrmm.modelgen import modules as MM

RAND_5_3_1_8_TENSOR = torch.tensor([[[[-39.5777,  37.3365,   1.4405, -45.2284, -30.2323,  -9.0451, -48.4779,
           -92.7080]],

         [[-14.7966,  89.5923,   7.5709,  -5.8178,  61.0096, -43.7362, -81.3752,
            77.7022]],

         [[ 29.9796, -59.5363, -27.8070,  91.9963,  81.7157,  48.2377,  17.6885,
            26.1734]]],


        [[[-68.9737, -70.6579, -23.2068,  38.3670, -94.0655, -83.8908,  38.5580,
             1.5770]],

         [[-25.0265, -49.9815,  69.5496,   3.4490,  51.3770,  62.3854, -38.0524,
           -47.5382]],

         [[ 81.9324,   4.6670,  42.3409,  72.8345,  72.6126,  14.7083, -54.6117,
            -2.2627]]],


        [[[ 74.2213, -43.8068,  32.0904, -17.9325, -75.6622, -90.8982,  54.6175,
            -8.4679]],

         [[-80.2494, -72.2750,  83.5952, -46.7447, -50.3009,  46.9418, -63.9820,
           -29.6908]],

         [[-73.8939,  93.0601,  26.8631,  50.2713, -49.2211, -38.0757,  50.9038,
            87.4260]]],


        [[[-98.4816,  58.3194,  26.5082,  96.8737,  82.4586, -83.9593,  85.2452,
           -99.9068]],

         [[ 95.4912,  19.7700, -51.6627,  -6.2033, -86.4973, -58.2238, -57.0516,
           -58.2348]],

         [[  5.4795,  73.6223, -74.7207, -98.4580, -14.4236,  42.8419,  90.8393,
            28.3862]]],


        [[[  5.1614,  27.7104, -26.2113, -97.0259, -21.3627,  54.6779,  25.2733,
            43.0408]],

         [[ 36.2706,  40.6934, -21.3570, -99.4077,  23.6329,  41.3776, -50.4784,
            45.5230]],

         [[ 78.7111,  60.3576,  97.1851,  19.8483,  72.0706, -12.9659, -87.3749,
           -66.8260]]]])

def test_ShallowRiskCtrlMLP_forward_0():
    """hard-code model weights and check forward pass"""
    # ~~~ ARRANGE ~~~
    n_inputs = 8
    n_ctrl_dims = 2
    n_neurons = 3
    model = MM.ShallowRiskCtrlMLP(n_inputs, n_ctrl_dims, n_neurons)

    # hard-code model weights
    with torch.no_grad():
        assert model.fc1.weight.shape == torch.Size((n_neurons, n_inputs))
        assert model.fc1.bias.shape == torch.Size((n_neurons,))
        model.fc1.weight[0] = nn.Parameter(0*torch.ones(1, n_inputs))
        model.fc1.weight[1] = nn.Parameter(1*torch.ones(1, n_inputs))
        model.fc1.weight[2] = nn.Parameter(2*torch.ones(1, n_inputs))
        model.fc1.bias = nn.Parameter(torch.zeros(n_neurons,))

        assert model.fc2.weight.shape == torch.Size((n_ctrl_dims+2, n_neurons))
        assert model.fc2.bias.shape == torch.Size((n_ctrl_dims+2,))
        model.fc2.weight[0] = nn.Parameter(1*torch.ones(1, n_neurons))
        model.fc2.weight[1] = nn.Parameter(2*torch.ones(1, n_neurons))
        model.fc2.weight[2] = nn.Parameter(3*torch.ones(1, n_neurons))
        model.fc2.weight[3] = nn.Parameter(4*torch.ones(1, n_neurons))
        model.fc2.bias = nn.Parameter(torch.zeros(n_ctrl_dims+2,))

    # specify inputs and compute expected outputs
    inp1 = torch.tensor([ 0.3561, -0.3459,  0.4423,  0.8048,  0.8366, -0.3073,  0.3673,  0.0882])
    out1_exp = torch.zeros(n_ctrl_dims+2,)
    hid1_1_0 = torch.tensor(0.)
    hid1_1_1 = F.elu(torch.sum(inp1))
    hid1_1_2 = F.elu(2*torch.sum(inp1))
    hid1_1 = torch.tensor([hid1_1_0, hid1_1_1, hid1_1_2])
    hid1_2_0 = torch.sum(hid1_1)
    hid1_2_1 = 2*torch.sum(hid1_1)
    hid1_2_2 = 3*torch.sum(hid1_1)
    hid1_2_3 = 4*torch.sum(hid1_1)
    out1_exp[0] = torch.sigmoid(hid1_2_0)
    out1_exp[1:] = torch.tensor([hid1_2_1, hid1_2_2, hid1_2_3])

    inp2 = torch.tensor([[[ 1.0022, -3.1395,  3.7320, -4.7389, -2.9087, -1.0383,  1.4305,0.9955]]])
    out2_exp = torch.zeros(1,1,n_ctrl_dims+2)
    hid2_1_0 = torch.zeros(1,1,1)
    hid2_1_1 = F.elu(torch.sum(inp2)).reshape(1,1,1)
    hid2_1_2 = F.elu(2*torch.sum(inp2)).reshape(1,1,1)
    hid2_1 = torch.cat([hid2_1_0, hid2_1_1, hid2_1_2], -1)
    hid2_2_0 = torch.sum(hid2_1).reshape(1,1,1)
    hid2_2_1 = 2*torch.sum(hid2_1).reshape(1,1,1)
    hid2_2_2 = 3*torch.sum(hid2_1).reshape(1,1,1)
    hid2_2_3 = 4*torch.sum(hid2_1).reshape(1,1,1)
    out2_exp[...,0] = torch.sigmoid(hid2_2_0).reshape(1,1,1)
    out2_exp[...,1:] = torch.cat([hid2_2_1, hid2_2_2, hid2_2_3], -1)

    inp3 = RAND_5_3_1_8_TENSOR
    out3_exp = torch.zeros(5,3,1,n_ctrl_dims+2)
    hid3_1_0 = torch.zeros(5,3,1,1)
    hid3_1_1 = F.elu(torch.sum(inp3,-1,True))
    hid3_1_2 = F.elu(2*torch.sum(inp3,-1,True))
    hid3_1 = torch.cat([hid3_1_0, hid3_1_1, hid3_1_2], -1)
    hid3_2_0 = torch.sum(hid3_1,-1,True)
    hid3_2_1 = 2*torch.sum(hid3_1,-1,True)
    hid3_2_2 = 3*torch.sum(hid3_1,-1,True)
    hid3_2_3 = 4*torch.sum(hid3_1,-1,True)
    out3_exp[...,0:1] = torch.sigmoid(hid3_2_0)
    out3_exp[...,1:] = torch.cat([hid3_2_1, hid3_2_2, hid3_2_3], -1)

    # ~~~ ACT ~~~
    # random-but-fixed input
    out1 = model(inp1)
    out2 = model(inp2)
    out3 = model(inp3)

    # ~~~ ASSERT ~~~
    assert out1.shape == torch.Size((n_ctrl_dims+2,))
    assert torch.allclose(out1, out1_exp)

    assert out2.shape == torch.Size((1,1,n_ctrl_dims+2))
    assert torch.allclose(out2, out2_exp)

    assert out3.shape == torch.Size((5,3,1,n_ctrl_dims+2))
    assert torch.allclose(out3, out3_exp)

if __name__ == "__main__":
    test_ShallowRiskCtrlMLP_forward_0()