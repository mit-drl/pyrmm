import faulthandler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from pyrmm.setups.dubins import DubinsPPMSetup

PPM_FILE = 'border_640x400.ppm'
SPEED = 10.0
MIN_TURN_RADIUS = 50.0

N_SAMPLES = 500

DURATION = 2.0
# N_BRANCHS = [4, 8]
# DEPTHS = [2, 4]
# N_BRANCHS = [1, 2]
# DEPTHS = [1, 2]
N_BRANCHS = [4]
DEPTHS = [6]
N_STEPS = 2

CMAP = 'coolwarm'


if __name__ == "__main__":
    # faulthandler.enable()

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

    # fig, axs = plt.subplots(2)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    for brch in N_BRANCHS:
        for dpth in DEPTHS:

            rmetrics = N_SAMPLES * [None]
            for i in range(N_SAMPLES):

                # evaluate risk metrics
                rmetrics[i] = dubss.estimateRiskMetric(
                    state=ssamples[i],
                    trajectory=None,
                    distance=DURATION,
                    branch_fact=brch,
                    depth=dpth,
                    n_steps=N_STEPS
                )

            # plot results
            lbl = "{}-branches, {}-depth".format(brch, dpth)
            xvals = [s.getX() for s in ssamples]
            yvals = [s.getY() for s in ssamples]
            uvals = [np.cos(s.getYaw()) for s in ssamples]
            vvals = [np.sin(s.getYaw()) for s in ssamples]
            # ax.quiver(xvals, yvals, uvals, vvals, rmetrics, cmap=CMAP, ec='k', lw=0.5)
            ax.quiver(xvals, yvals, uvals, vvals, rmetrics, cmap=CMAP)
            # ax.plot(xsamples, rmetrics, '*', label=lbl)

    # Draw in true risk metrics and obstacles
    # ax.axhline(y=1.0, label='Random Policy True Risk', ls='-', c='r')
    # ax.axhline(y=0.0, label='Optimal Policy True Risk', ls='-', c='b')
    ax.axvline(x=0, label='Obstacle'.format(0.5), lw=4.0, c='k')
    ax.axvline(x=640, label='Obstacle'.format(0.5), lw=4.0, c='k')
    ax.axhline(y=0, label='Obstacle'.format(0.5), lw=4.0, c='k')
    ax.axhline(y=400, label='Obstacle'.format(0.5), lw=4.0, c='k')
    # ax.axvline(x=1.0, label='Obstacle'.format(0.5), lw=4.0, c='k')
    # handles, labels = ax.get_legend_handles_labels()
    # lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05,0.5))
    ax.set_title(
        "Estimated Risk Metrics for Dubins Vehicle in Constrained Box\n"+
        "w/ uniform control sampling of duration={},\n".format(DURATION) +
        "tree depth={}, and branching factor={}".format(DEPTHS[0], N_BRANCHS[0]) 
    )
    ax.set_xlim([0,640])
    ax.set_ylim([0,400])
    ax.set_xlabel("x-position")
    ax.set_ylabel("y-position")
    fig.colorbar(cm.ScalarMappable(None, CMAP), ax=ax, label='failure probability')
    # fig.savefig('dubins_risk_estimation', bbox_extra_artists=(lgd,), bbox_inches='tight')
    fig.savefig('dubins_risk_estimation', bbox_inches='tight')
    # plt.show()