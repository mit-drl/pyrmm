# Example of a risk-metric control barrier function
# for 1D Double integrator System

import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools

from functools import partial

import pyrmm.utils.utils as U
from pyrmm.environments.dubins4d_reachavoid import cvx_qp_solver

from pyrmm.setups.double_integrator import DoubleIntegrator1DSetup

from pyrmm.modelgen.data_modules import LSFORDataModule
from pyrmm.modelgen.modules import ShallowRiskCBFPerceptron, CBFLRMMModule, DeepRiskCBFPerceptron
from pyrmm.modelgen.double_integrator import quadratic_state_feature_map, local_states_datagen, trivial_state_feature_map

###
# Problem parameters
###

# obstacle location on x-axis
P_OBS = 5.0 # [m]

# control bounds
U1_MIN = -1.0   # [m/s/s]
U1_MAX = 1.0    # [m/s/s]

# evaluate derivatives at current state, i.e. local origin
XTIL_1 = 0.0
XTIL_2 = 0.0

# initial/current state of system, or 
# state at which observation z is taken
P_Z_ARR = np.linspace(-1.0, 4.5, 20)    # [m]
# V_Z_ARR = np.arange(-1, 2.5, 0.5)   # [m/s]
V_Z_ARR = np.linspace(-1, 2.5, 20)   # [m/s]

# plotting params
MARKER = itertools.cycle((',', '+', '.', 'o', '*')) 

def run_analytical_cbf_analysis():

    results = np.empty((len(V_Z_ARR), len(P_Z_ARR)))

    for i, v_z in enumerate(V_Z_ARR):
        for j, p_z in enumerate(P_Z_ARR):

            ###
            # Computed parameters
            ###

            # control input for minimum stopping distance
            if v_z >= 0:
                u1_minstop = U1_MIN
            else:
                u1_minstop = U1_MAX

            # compute final position of minimum-distance stopping
            p_minstop = -0.5 * v_z**2 / u1_minstop + p_z

            # break if stopping impossible
            if p_minstop > P_OBS:
                results[i,j] = np.nan
                continue

            # risk metric at current state
            rho_x = np.exp(-(P_OBS - p_minstop))

            # barrier function value at current state
            h_x = 1 - rho_x

            ###
            # Compute Lie derivatives
            ###

            # Lie derivative along g(x)
            Lghx1 = ((XTIL_2 + v_z) / u1_minstop) * np.exp(p_minstop - P_OBS)

            # Lie derivative along f(x)
            Lfhx = -(XTIL_2 + v_z) * np.exp(p_minstop - P_OBS)

            ### 
            # Formulate QP
            ###

            # objective: minimize control input
            P = np.eye(1).reshape(1,1)
            q = np.zeros(1).reshape(1,1)
            lambda_h = 0.1

            # Constraint: Left side of <= inequality
            G = np.reshape([-Lghx1], (1,1))

            # Constraint: right side of <= inequality
            h = np.reshape([Lfhx + lambda_h * h_x], (1,1))

            ### 
            # solve QP
            ###

            u_opt = cvx_qp_solver(P=P, q=q, G=G, h=h)
            results[i,j] = u_opt[0]

            # print("Min-norm safe control = ", u_opt[0])

        plt.plot(P_Z_ARR, results[i], marker = next(MARKER), label="v={} m/s".format(v_z))

    plt.title("Safe Control Input As Function of Position and Velocity")
    plt.xlabel("position [m]")
    plt.ylabel("control input [m/s/s]")
    plt.legend()
    plt.show()

# create model object to load saved model into
# Note: this is hacky to have to remember the exact number of observation
# inputs, state features, and neurons. There has got to be a better way 
# save these params at model save time
n_obsv_dim = 3
n_feat_dim = 6 
# n_feat_dim = 2
# n_neurons = 8
# rmcbf_model = ShallowRiskCBFPerceptron(
#     num_obs_inputs=n_obsv_dim,
#     num_state_features=n_feat_dim,
#     num_neurons=n_neurons
# )
rmcbf_model = DeepRiskCBFPerceptron(
    num_obs_inputs=n_obsv_dim,
    num_state_features=n_feat_dim,
    num_neurons=[32, 16, 8]
)

# load checkpoint
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     "outputs/2023-01-19/12-34-03/lightning_logs/version_0/" +
#     "checkpoints/epoch=511-step=308735.ckpt"
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     "outputs/2023-01-20/15-35-25/lightning_logs/version_0/" +
#     "checkpoints/epoch=511-step=308735.ckpt"
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     "outputs/2023-02-03/13-43-20/lightning_logs/version_0/checkpoints/epoch=511-step=308735.ckpt"
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     'outputs/2023-02-07/08-08-31/lightning_logs/version_0/checkpoints/epoch=284-step=2125529.ckpt'
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     'outputs/2023-02-07/12-49-53/lightning_logs/version_0/epoch=71-step=1339919.ckpt'
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     'outputs/2023-02-07/15-02-24/lightning_logs/version_0/epoch=263-step=1240535.ckpt'
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     'outputs/2023-02-07/16-39-56/lightning_logs/version_0/checkpoints/epoch=247-step=1111287.ckpt'
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     'outputs/2023-02-07/19-08-10/lightning_logs/version_0/epoch=123-step=555643.ckpt'
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     'outputs/2023-02-07/20-59-56/lightning_logs/version_0/epoch=66-step=300226.ckpt'
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     'outputs/2023-02-07/21-36-53/lightning_logs/version_0/epoch=73-step=347725.ckpt'
# )
# chkpt_file = (
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     'outputs/2023-02-07/22-48-57/lightning_logs/version_0/epoch=118-step=501822.ckpt'
# )
chkpt_file = (
    "/home/ross/Projects/AIIA/risk_metric_maps/" +
    'outputs/2023-02-08/07-24-31/lightning_logs/version_0/epoch=41-step=781619.ckpt'
)
chkpt = torch.load(chkpt_file)

# create pytorch lightning module
# no optimizer because this is only for eval/inference
rmcbf = CBFLRMMModule(num_inputs=3, model=rmcbf_model, optimizer=None)

# load checkpoint into module
rmcbf.load_state_dict(chkpt['state_dict'])
rmcbf.eval()

# create data module and load training data to get observation scaler
# datapaths = U.get_abs_pt_data_paths(
#     "/home/ross/Projects/AIIA/risk_metric_maps/" +
#     "outputs/2022-12-19/14-09-56/"
# )
datapaths = U.get_abs_pt_data_paths(
    "/home/ross/Projects/AIIA/risk_metric_maps/" +
    "outputs/2023-02-03/15-30-08/"
)

# local_states_datagen_func = partial(local_states_datagen, 1.5)  # again, this is just something you need to know was used during modelgen
# local_states_datagen_func = partial(local_states_datagen, 3)  # again, this is just something you need to know was used during modelgen
local_states_datagen_func = partial(local_states_datagen, 1.0)  # again, this is just something you need to know was used during modelgen
rmcbf_data_mod = LSFORDataModule(
    datapaths=datapaths,
    val_ratio=0,
    batch_size=1,
    num_workers=0,
    state_feature_map=quadratic_state_feature_map,  # again this is something you just need to know was used during modelgen, very hacky/brittle to encode this way
    # state_feature_map=trivial_state_feature_map,  # again this is something you just need to know was used during modelgen, very hacky/brittle to encode this way
    local_states_datagen=local_states_datagen_func,
    compile_verify_func=None
)
rmcbf_data_mod.setup("test")

# create double integrator setup to query proper observations
pos_bounds = [-32, 32]
vel_bounds = [-8, 8]
acc_bounds = [U1_MIN, U1_MAX] # deterministic system with no accel
obst_bounds = [P_OBS, P_OBS+0.1]    # very slim obstacle
di1d_setup = DoubleIntegrator1DSetup(
    pos_bounds=pos_bounds, 
    vel_bounds=vel_bounds, 
    acc_bounds=acc_bounds, 
    obst_bounds=obst_bounds)

def run_irbf_global_analysis():

    # iterate through states
    risks = np.empty((len(V_Z_ARR), len(P_Z_ARR)))
    cbf_weights = np.empty((len(V_Z_ARR), len(P_Z_ARR), n_feat_dim))
    ctrls = np.empty((len(V_Z_ARR), len(P_Z_ARR)))
    for i, v_z in enumerate(V_Z_ARR):
        for j, p_z in enumerate(P_Z_ARR):

            print("DEBUG: i,j = ",i,j)

            # package state in np array and convert state to ompl
            s_z_np = np.array([p_z, v_z])
            s_z_ompl = di1d_setup.space_info.allocState()
            di1d_setup.state_numpy_to_ompl(np_state=s_z_np, omplState=s_z_ompl)
            print("DEBUG: state = ",s_z_np)

            # get observation of state
            o_z_np = di1d_setup.observeState(state=s_z_ompl)
            print("DEBUG: observation = ",o_z_np)

            # scale observation
            o_z_scaled_pt = torch.from_numpy(
                np.float32(
                    rmcbf_data_mod.observation_scaler.transform(
                        o_z_np.reshape(1,n_obsv_dim)
                    )
                )
            ).reshape(n_obsv_dim,)
            print("DEBUG: scaled observation = ",o_z_scaled_pt)

            # compute state features of localized state
            stil_z_np = np.array([XTIL_1, XTIL_2])
            ftil_z_np = rmcbf_data_mod.state_feature_map(stil_z_np)
            print("DEBUG: state feature of local state = ", ftil_z_np)

            # infer risk metric and output cbf weights from trained model
            rho_z_pt, w_cbf_pt = rmcbf.forward(observation=o_z_scaled_pt, state_features=torch.from_numpy(ftil_z_np))
            risks[i,j] = rho_z_pt.detach().numpy()
            cbf_weights[i,j,:] = w_cbf_pt.detach().numpy()
            print("DEBUG: cbf weights = ", cbf_weights[i,j,:])
            print("DEBUG: inferred risk metric = ", rho_z_pt)

        plt.plot(P_Z_ARR, risks[i], marker = next(MARKER), label="v={} m/s".format(v_z))

    plt.title("Risk Estimate As Function of Position and Velocity\n1-D Double Integrator System with Obstacle at position=5")
    plt.xlabel("position [m]")
    plt.ylabel("probability of failure [-]")
    plt.legend()
    plt.show()


    p_mesh, v_mesh = np.meshgrid(P_Z_ARR, V_Z_ARR)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(p_mesh, v_mesh, risks, cmap="turbo", vmin=0, vmax=1)
    fig.colorbar(pcm, ax=ax)
    plt.title("Risk Estimate (Failure Probability) Inferred at Each State\n"+
        "for 1-D Double Integrator System with Uniform Random Policy and\n"+
        "Obstacle at position=5")
    plt.xlabel("position [m]")
    plt.ylabel("velocity [m/s]")
    plt.show()

    #################################

def run_irbf_local_analysis():

    # analyze inferred risk function from single observation
    # package state in np array and convert state to ompl
    # s_z_np = np.zeros(2)
    # s_z_np = np.array([4.5, 0.0])
    # s_z_np = np.array([4.5, 2.0])
    # s_z_np = np.array([2.0, 1.5])
    s_z_np = np.array([3.0, 1.0])
    # s_z_np = np.array([3.0, 2.0])
    s_z_ompl = di1d_setup.space_info.allocState()
    di1d_setup.state_numpy_to_ompl(np_state=s_z_np, omplState=s_z_ompl)
    print("DEBUG: state = ",s_z_np)

    # get observation of state
    o_z_np = di1d_setup.observeState(state=s_z_ompl)
    print("DEBUG: observation = ",o_z_np)

    # scale observation
    o_z_scaled_pt = torch.from_numpy(
        np.float32(
            rmcbf_data_mod.observation_scaler.transform(
                o_z_np.reshape(1,n_obsv_dim)
            )
        )
    ).reshape(n_obsv_dim,)
    print("DEBUG: scaled observation = ",o_z_scaled_pt)

    # iterate through states
    risks_0_0 = np.empty((len(V_Z_ARR), len(P_Z_ARR)))
    # stil_z_np = np.empty((len(V_Z_ARR), len(P_Z_ARR), 2))
    for i, v_z in enumerate(V_Z_ARR):
        for j, p_z in enumerate(P_Z_ARR):

            # compute state features of localized state
            stil_z_np = np.array([p_z, v_z]) - s_z_np
            ftil_z_np = rmcbf_data_mod.state_feature_map(stil_z_np)
            print("DEBUG: state feature of local state = ", ftil_z_np)

            # infer risk metric and output cbf weights from trained model
            rho_z_pt, w_cbf_pt = rmcbf.forward(observation=o_z_scaled_pt, state_features=torch.from_numpy(ftil_z_np))
            risks_0_0[i,j] = rho_z_pt.detach().numpy()
            # cbf_weights[i,j,:] = w_cbf_pt.detach().numpy()
            # print("DEBUG: cbf weights = ", cbf_weights[i,j,:])
            print("DEBUG: cbf weights = ", w_cbf_pt.detach().numpy())

        # plt.plot(P_Z_ARR, risks_0_0[i], marker = next(MARKER), label="v={} m/s".format(v_z))

    # ref: https://stackoverflow.com/a/54088910/4055705
    p_mesh, v_mesh = np.meshgrid(P_Z_ARR, V_Z_ARR)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(p_mesh, v_mesh, risks_0_0, cmap="turbo", vmin=0, vmax=1)
    fig.colorbar(pcm, ax=ax)
    plt.plot(s_z_np[0], s_z_np[1], 'k+', ms=10, mew=3)
    plt.title("Risk Estimate (Failure Probability) Extrapolated from\n"+
        "Single-Point Risk Function at p={} [m], v= {} [m/s]\n".format(s_z_np[0], s_z_np[1])+
        "for 1-D Double Integrator System with\n" + 
        "Uniform Random Policy and Obstacle at position=5")
    plt.xlabel("position [m]")
    plt.ylabel("velocity [m/s]")
    plt.show()

    # plt.title("Risk Estimate As Function of Position and Velocity\nfrom Single Point Risk Function\n1-D Double Integrator System with Obstacle at position=5")
    # plt.xlabel("position [m]")
    # plt.ylabel("probability of failure [-]")
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    # run_analytical_cbf_analysis()
    run_irbf_global_analysis()
    run_irbf_local_analysis()
