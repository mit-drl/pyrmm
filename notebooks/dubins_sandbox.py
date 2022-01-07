import faulthandler

# import pickle
# import dill
# pickle.Pickler = dill.Pickler
import copyreg
import argparse

# import dill as pickle
from joblib import Parallel, delayed
import multiprocess
from functools import partial
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from ompl import base as ob
from pyrmm.setups.dubins import DubinsPPMSetup

PPM_FILE = 'border_640x400.ppm'
SPEED = 10.0
MIN_TURN_RADIUS = 50.0

N_SAMPLES = 20

DURATION = 2.0
# N_BRANCHS = [4, 8]
# DEPTHS = [2, 4]
# N_BRANCHS = [1, 2]
# DEPTHS = [1, 2]
N_BRANCHS = [8]
DEPTHS = [4]
N_STEPS = 2

CMAP = 'coolwarm'

# RUN_PARALLEL = False

SE2SPACE = ob.SE2StateSpace()

def _pickle_SE2StateInternal(state):
    x = state.getX()
    y = state.getY()
    yaw = state.getYaw()
    return _unpickle_SE2StateInternal, (x, y, yaw)

def _unpickle_SE2StateInternal(x, y, yaw):
    state = SE2SPACE.allocState()
    state.setX(x)
    state.setY(y)
    state.setYaw(yaw)
    return state

copyreg.pickle(SE2SPACE.SE2StateInternal, _pickle_SE2StateInternal, _unpickle_SE2StateInternal)

parser = argparse.ArgumentParser()
parser.add_argument("--run-parallel",action='store_true')

if __name__ == "__main__":
    # faulthandler.enable()

    args = parser.parse_args()

    # measure overhead init time
    t0 = time.time()

    # create single integrator system
    fpath = str(Path(__file__).parent.resolve().joinpath(PPM_FILE))
    dubss = DubinsPPMSetup(
        ppm_file=fpath,
        speed=SPEED,
        min_turn_radius=MIN_TURN_RADIUS
    )

    # sample states to evaluate risk metrics
    sampler = dubss.space_info.allocValidStateSampler()
    ssamples = N_SAMPLES * [None] 
    for i in range(N_SAMPLES):

        # assign state
        ssamples[i] = dubss.space_info.allocState()
        sampler.sample(ssamples[i])

    t1 = time.time()
    if args.run_parallel:
        num_cores = multiprocess.cpu_count()

        # multiprocess implementation
        partial_estimateRiskMetric = partial(dubss.estimateRiskMetric, 
            trajectory=None,
            distance=DURATION,
            branch_fact=N_BRANCHS[0],
            depth=DEPTHS[0],
            n_steps=N_STEPS
        )
        with multiprocess.Pool(num_cores) as pool:
            rmetrics = pool.map(partial_estimateRiskMetric, ssamples)

        # joblib implementation
        # rmetrics = Parallel(n_jobs=num_cores)(
        #     delayed(dubss.estimateRiskMetric)(
        #         state=s,
        #         trajectory=None,
        #         distance=DURATION,
        #         branch_fact=N_BRANCHS[0],
        #         depth=DEPTHS[0],
        #         n_steps=N_STEPS
        #     ) for s in ssamples
        # )

    else:
        times = np.asarray((N_SAMPLES+1) * [None])
        rmetrics = N_SAMPLES * [None]
        for i in range(N_SAMPLES):
            times[i] = time.time()

            # evaluate risk metrics
            rmetrics[i] = dubss.estimateRiskMetric(
                state=ssamples[i],
                trajectory=None,
                distance=DURATION,
                branch_fact=N_BRANCHS[0],
                depth=DEPTHS[0],
                n_steps=N_STEPS
            )

        times[-1] = time.time()
        t_calc_avg = np.mean(times[1:] - times[:-1])
        print("average risk metric calculation time ", t_calc_avg)

    print("overhead initialization time ", t1 - t0)
    print("total time ", time.time()-t0)

    # plot results
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    # lbl = "{}-branches, {}-depth".format(brch, dpth)
    xvals = [s.getX() for s in ssamples]
    yvals = [s.getY() for s in ssamples]
    uvals = [np.cos(s.getYaw()) for s in ssamples]
    vvals = [np.sin(s.getYaw()) for s in ssamples]
    ax.quiver(xvals, yvals, uvals, vvals, rmetrics, cmap=CMAP)

    # Draw in true risk metrics and obstacles
    ax.axvline(x=0, label='Obstacle'.format(0.5), lw=4.0, c='k')
    ax.axvline(x=640, label='Obstacle'.format(0.5), lw=4.0, c='k')
    ax.axhline(y=0, label='Obstacle'.format(0.5), lw=4.0, c='k')
    ax.axhline(y=400, label='Obstacle'.format(0.5), lw=4.0, c='k')
    ax.set_title(
        "Estimated Risk Metrics for Dubins Vehicle (speed={}, turn rad={})\n".format(SPEED, MIN_TURN_RADIUS) +
        "in Constrained Box w/ uniform control sampling of duration={},\n".format(DURATION) +
        "tree depth={}, and branching factor={}".format(DEPTHS[0], N_BRANCHS[0]) 
    )
    ax.set_xlim([0,640])
    ax.set_ylim([0,400])
    ax.set_xlabel("x-position")
    ax.set_ylabel("y-position")
    fig.colorbar(cm.ScalarMappable(None, CMAP), ax=ax, label='failure probability')
    fig.savefig('dubins_risk_estimation', bbox_inches='tight')
    plt.show()