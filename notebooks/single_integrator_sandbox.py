import faulthandler
import numpy as np
import matplotlib.pyplot as plt

from pyrmm.setups.single_integrator import SingleIntegrator1DSetup

V_LOW = 0.0
V_HIGH = 1.0
LBOUND = 0.0
UBOUND = 1.0
N_SAMPLES = 100

DURATION = 0.5
N_BRANCHS = [4, 8]
DEPTHS = [2, 4]
# N_BRANCHS = [1, 2]
# DEPTHS = [1, 2]
N_STEPS = 2


if __name__ == "__main__":
    # faulthandler.enable()

    # create single integrator system
    span = UBOUND - LBOUND
    siss = SingleIntegrator1DSetup(
        min_speed=V_LOW,
        max_speed=V_HIGH, 
        lower_bound=LBOUND, 
        upper_bound=UBOUND
    )

    # sample states to evaluate risk metrics
    xsamples = np.random.rand(N_SAMPLES) * span + LBOUND
    ssamples = N_SAMPLES * [None] 
    for i in range(N_SAMPLES):

        # assign state
        ssamples[i] = siss.space_info.allocState()
        ssamples[i][0] = xsamples[i]

    # fig, axs = plt.subplots(2)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    for brch in N_BRANCHS:
        for dpth in DEPTHS:

            rmetrics = N_SAMPLES * [None]
            for i in range(N_SAMPLES):

                # evaluate risk metrics
                rmetrics[i] = siss.estimateRiskMetric(
                    state=ssamples[i],
                    trajectory=None,
                    distance=DURATION,
                    branch_fact=brch,
                    depth=dpth,
                    n_steps=N_STEPS
                )

            # plot results
            lbl = "{}-branches, {}-depth".format(brch, dpth)
            ax.plot(xsamples, rmetrics, '*', label=lbl)

    # Draw in true risk metrics and obstacles
    ax.axhline(y=1.0, label='Random Policy True Risk', ls='-', c='r')
    ax.axhline(y=0.0, label='Optimal Policy True Risk', ls='-', c='b')
    ax.axvline(x=1.0, label='Obstacle'.format(0.5), lw=4.0, c='k')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05,0.5))
    ax.set_title(
        "Estimated Risk Metrics for 1D, Positive-Velocity, Single Integrator\n"+
        "w/ uniform control sampling of duration {}".format(DURATION) 
    )
    ax.set_xlabel("state position")
    ax.set_ylabel("estimated risk metric")
    fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')