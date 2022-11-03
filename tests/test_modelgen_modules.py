import torch
import torch.nn as nn
import torch.nn.functional as F
from pyrmm.modelgen import modules as MM

RAND_4_2_6_1_4_TENSOR = torch.tensor([[[[[0.1677, 0.2184, 0.3521, 0.9554]],

          [[0.8713, 0.2199, 0.3058, 0.4322]],

          [[0.7881, 0.7461, 0.6340, 0.7941]],

          [[0.6209, 0.6244, 0.0467, 0.4017]],

          [[0.6277, 0.6336, 0.8452, 0.9840]],

          [[0.7993, 0.1049, 0.8114, 0.5344]]],


         [[[0.5524, 0.8691, 0.6199, 0.6603]],

          [[0.6554, 0.9292, 0.1633, 0.2750]],

          [[0.8987, 0.2399, 0.9011, 0.3404]],

          [[0.4862, 0.7017, 0.8860, 0.3614]],

          [[0.0944, 0.1052, 0.1760, 0.6474]],

          [[0.8539, 0.4215, 0.2557, 0.2822]]]],



        [[[[0.8443, 0.0318, 0.7286, 0.0715]],

          [[0.4439, 0.4007, 0.8850, 0.2651]],

          [[0.4317, 0.5953, 0.8676, 0.1578]],

          [[0.0746, 0.1041, 0.5948, 0.4888]],

          [[0.8641, 0.3693, 0.4515, 0.6164]],

          [[0.4418, 0.9042, 0.2875, 0.4966]]],


         [[[0.9613, 0.2539, 0.0026, 0.9562]],

          [[0.7217, 0.5346, 0.7114, 0.6980]],

          [[0.7859, 0.7069, 0.9861, 0.3226]],

          [[0.3678, 0.5340, 0.6261, 0.9519]],

          [[0.4955, 0.4273, 0.6983, 0.6343]],

          [[0.5089, 0.4279, 0.4698, 0.9542]]]],



        [[[[0.4666, 0.1278, 0.5355, 0.8196]],

          [[0.8455, 0.0176, 0.1363, 0.9741]],

          [[0.6919, 0.3461, 0.0039, 0.2900]],

          [[0.1520, 0.4935, 0.6485, 0.0219]],

          [[0.6051, 0.4294, 0.9639, 0.6025]],

          [[0.7989, 0.2177, 0.0391, 0.6560]]],


         [[[0.2296, 0.7617, 0.6791, 0.5682]],

          [[0.9452, 0.9602, 0.4999, 0.1953]],

          [[0.2267, 0.8846, 0.8819, 0.7731]],

          [[0.6824, 0.4882, 0.4015, 0.2992]],

          [[0.2108, 0.1426, 0.4578, 0.9106]],

          [[0.4775, 0.9816, 0.7372, 0.7029]]]],



        [[[[0.1354, 0.5455, 0.9623, 0.8702]],

          [[0.2656, 0.1996, 0.3328, 0.3414]],

          [[0.7173, 0.8497, 0.9758, 0.4026]],

          [[0.2960, 0.6937, 0.8111, 0.4635]],

          [[0.6926, 0.4000, 0.0802, 0.3729]],

          [[0.5962, 0.0054, 0.9527, 0.7220]]],


         [[[0.1013, 0.2954, 0.9199, 0.3937]],

          [[0.0366, 0.9732, 0.2806, 0.7713]],

          [[0.9291, 0.7528, 0.7275, 0.7628]],

          [[0.3431, 0.5089, 0.3752, 0.1526]],

          [[0.8311, 0.5873, 0.6990, 0.4195]],

          [[0.7042, 0.2247, 0.4263, 0.2764]]]]])

RAND_4_2_6_1_2_TENSOR = torch.tensor([[[[[0.4108, 0.2092]],

          [[0.5327, 0.7215]],

          [[0.9850, 0.9995]],

          [[0.7537, 0.7985]],

          [[0.1308, 0.6763]],

          [[0.0756, 0.7314]]],


         [[[0.8545, 0.1918]],

          [[0.2089, 0.6209]],

          [[0.1960, 0.8911]],

          [[0.2316, 0.2038]],

          [[0.3577, 0.8457]],

          [[0.8330, 0.4900]]]],



        [[[[0.5825, 0.6537]],

          [[0.3015, 0.2786]],

          [[0.6521, 0.7614]],

          [[0.7565, 0.8416]],

          [[0.9662, 0.8570]],

          [[0.3999, 0.1864]]],


         [[[0.0945, 0.9571]],

          [[0.8643, 0.3071]],

          [[0.8637, 0.6851]],

          [[0.9470, 0.4553]],

          [[0.3119, 0.0448]],

          [[0.6461, 0.3741]]]],



        [[[[0.2062, 0.3471]],

          [[0.6397, 0.7517]],

          [[0.9197, 0.7897]],

          [[0.1690, 0.7011]],

          [[0.2386, 0.0899]],

          [[0.6867, 0.6877]]],


         [[[0.2910, 0.2923]],

          [[0.8074, 0.2456]],

          [[0.7153, 0.8692]],

          [[0.9027, 0.4136]],

          [[0.0253, 0.2783]],

          [[0.3872, 0.5154]]]],



        [[[[0.6981, 0.3272]],

          [[0.1124, 0.1985]],

          [[0.0980, 0.0126]],

          [[0.3133, 0.8688]],

          [[0.8632, 0.1329]],

          [[0.5567, 0.4747]]],


         [[[0.0423, 0.3334]],

          [[0.9349, 0.3737]],

          [[0.8551, 0.6653]],

          [[0.2583, 0.5326]],

          [[0.9217, 0.5123]],

          [[0.5111, 0.1277]]]]])



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

def test_ShallowRiskCBFPerceptron_forward_0():
    """hard-code model weights and check forward pass"""
    # ~~~ ARRANGE ~~~
    n_inputs = 4
    n_features = 2
    n_neurons = 3
    model = MM.ShallowRiskCBFPerceptron(n_inputs, n_features, n_neurons)

    # hard-code model weights
    with torch.no_grad():
        assert model.fc1.weight.shape == torch.Size((n_neurons, n_inputs))
        assert model.fc1.bias.shape == torch.Size((n_neurons,))
        model.fc1.weight[0] = nn.Parameter(0*torch.ones(1, n_inputs))
        model.fc1.weight[1] = nn.Parameter(1*torch.ones(1, n_inputs))
        model.fc1.weight[2] = nn.Parameter(2*torch.ones(1, n_inputs))
        model.fc1.bias = nn.Parameter(torch.zeros(n_neurons,))

        assert model.fc2.weight.shape == torch.Size((n_features, n_neurons))
        assert model.fc2.bias.shape == torch.Size((n_features,))
        model.fc2.weight[0] = nn.Parameter(1*torch.ones(1, n_neurons))
        model.fc2.weight[1] = nn.Parameter(2*torch.ones(1, n_neurons))
        model.fc2.bias = nn.Parameter(torch.zeros(n_features,))

    # specify inputs and compute expected outputs
    inp1 = torch.tensor([0.9605, 0.1075, 0.3812, 0.8606])
    feat1 = torch.tensor([0.9042, 0.4913])
    hid1_1_0 = torch.sigmoid(0*torch.sum(inp1))
    hid1_1_1 = torch.sigmoid(torch.sum(inp1))
    hid1_1_2 = torch.sigmoid(2*torch.sum(inp1))
    hid1_1 = torch.tensor([hid1_1_0, hid1_1_1, hid1_1_2])
    hid1_2_0 = torch.sum(hid1_1)
    hid1_2_1 = 2*torch.sum(hid1_1)
    w1_exp = torch.tensor([hid1_2_0, hid1_2_1])
    rho1_exp = torch.sigmoid(torch.dot(w1_exp, feat1))

    inp2 = torch.tensor([[[[0.5801, 0.5049, 0.3601, 0.9902]]]])
    feat2 = torch.tensor([[[[0.5711, 0.3319]]]])
    hid2_1_0 = torch.sigmoid(0*torch.sum(inp2))
    hid2_1_1 = torch.sigmoid(torch.sum(inp2))
    hid2_1_2 = torch.sigmoid(2*torch.sum(inp2))
    hid2_1 = torch.tensor([hid2_1_0, hid2_1_1, hid2_1_2])
    hid2_2_0 = torch.sum(hid2_1)
    hid2_2_1 = 2*torch.sum(hid2_1)
    w2_exp = torch.tensor([[[[hid2_2_0, hid2_2_1]]]])
    rho2_exp = torch.sigmoid(torch.inner(w2_exp, feat2))

    inp3 = RAND_4_2_6_1_4_TENSOR
    feat3 = RAND_4_2_6_1_2_TENSOR
    hid3_1_0 = torch.sigmoid(0*torch.sum(inp3,-1,True))
    hid3_1_1 = torch.sigmoid(torch.sum(inp3,-1,True))
    hid3_1_2 = torch.sigmoid(2*torch.sum(inp3,-1,True))
    hid3_1 = torch.cat([hid3_1_0, hid3_1_1, hid3_1_2], -1)
    hid3_2_0 = torch.sum(hid3_1,-1,True)
    hid3_2_1 = 2*torch.sum(hid3_1,-1,True)
    w3_exp = torch.cat([hid3_2_0, hid3_2_1],-1)
    rho3_exp = torch.sigmoid((w3_exp * feat3).sum(-1,keepdim=True))

    # ~~~ ACT ~~~
    # random-but-fixed input
    rho1, w1 = model(inp1, feat1)
    rho2, w2 = model(inp2, feat2)
    rho3, w3 = model(inp3, feat3)

    # ~~~ ASSERT ~~~
    assert rho1.shape == torch.Size((1,))
    assert w1.shape == torch.Size((n_features,))
    assert torch.allclose(w1, w1_exp)
    assert torch.allclose(rho1, rho1_exp)

    assert rho2.shape == torch.Size((1,1,1,1))
    assert w2.shape == torch.Size((1,1,1,n_features))
    assert torch.allclose(w2, w2_exp)
    assert torch.allclose(rho2, rho2_exp)

    assert rho3.shape == torch.Size((4,2,6,1,1))
    assert w3.shape == torch.Size((4,2,6,1,n_features))
    assert torch.allclose(w3, w3_exp)
    assert torch.allclose(rho3, rho3_exp)


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
    test_ShallowRiskCBFPerceptron_forward_0()
    # test_ShallowRiskCtrlMLP_forward_0()