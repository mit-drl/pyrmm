'''
a simple example using ompl's dubins vehicle

Ref: 
+ https://ompl.kavrakilab.org/python.html
+ https://ompl.kavrakilab.org/RigidBodyPlanningWithODESolverAndControls_8py_source.html
+ https://ompl.kavrakilab.org/Point2DPlanning_8py_source.html
'''

import pathlib
import dubins
import numpy as np

from copy import deepcopy
from functools import partial
from matplotlib import pyplot as plt

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
 
# def isStateValid(state):
#     # "state" is of type SE2StateInternal, so we don't need to use the "()"
#     # operator.
#     #
#     # Some arbitrary condition on the state (note that thanks to
#     # dynamic type checking we can just call getX() and do not need
#     # to convert state to an SE2State.)
#     return state.getX() < .51
 
# def plan():
#     # create an SE2 state space
#     space = ob.DubinsStateSpace()
 
#     # set lower and upper bounds
#     bounds = ob.RealVectorBounds(2)
#     bounds.setLow(-1)
#     bounds.setHigh(1)
#     space.setBounds(bounds)
 
#     # create a simple setup object
#     ss = og.SimpleSetup(space)
#     ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
 
#     start = ob.State(space)
#     # we can pick a random start state...
#     start.random()
#     # ... or set specific values
#     start().setX(.5)
#     start().setY(0)
#     start().setYaw(0)
 
#     goal = ob.State(space)
#     # we can pick a random goal state...
#     goal.random()
#     # ... or set specific values
#     goal().setX(.4)
#     goal().setY(0)
#     goal().setYaw(0)
 
#     ss.setStartAndGoalStates(start, goal)
 
#     # this will automatically choose a default planner with
#     # default parameters
#     solved = ss.solve(1.0)
 
#     if solved:
#         # try to shorten the path
#         ss.simplifySolution()
#         # print the simplified path
#         print(ss.getSolutionPath())

class DubinsTrajectorySegment:
    def __init__(self, alpha, beta, path):
        ''' 
        Args:
            alpha: ob.State
                initial state of segment
            beta: ob.State
                final state of segment
            path: dubins._DubinsPath
        '''
        self.alpha = alpha
        self.beta = beta
        self.path = path

    def discrete_path(self, step_size):
        '''discretize the path into States
        
        Args:
            step_size : float
                step size to use in dubins.path.sample_many
        
        Returns:
            disc_path : list(ob.State)
                list of discrete states
        '''
        cfgs, _ = self.path.sample_many(step_size)
        n_steps = len(cfgs)
        disc_path = n_steps*[None]
        for i in range(n_steps):
            # s = deepcopy(self.alpha)
            s = ob.State(ob.DubinsStateSpace())
            s().setX(cfgs[i][0])
            s().setY(cfgs[i][1])
            s().setYaw(cfgs[i][2])
            disc_path[i] = s

        return disc_path

class DubinsEnvironment:
    def __init__(self, ppm_file, turn_rad):
        '''
        Args:
            ppm_file: str
                file path to ppm image used as environment map
            turn_rad: float
                turning radius of the dubins vehicle
        '''
        self.ppm_file = ppm_file
        self.ppm = ou.PPM()
        self.ppm.loadFile(self.ppm_file)
        self.rho = turn_rad
        space = ob.DubinsStateSpace(turningRadius=self.rho)
        self.maxWidth = self.ppm.getWidth()-1
        self.maxHeight = self.ppm.getHeight()-1

        # set lower and upper bounds
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, 0.0)
        bounds.setHigh(0, self.ppm.getWidth())
        bounds.setLow(1, 0.0)
        bounds.setHigh(1, self.ppm.getHeight())
        space.setBounds(bounds)

        # create simple setup
        self.ss = og.SimpleSetup(space)

        # setup state validity checker (collision checker)
        self.ss.setStateValidityChecker(
            ob.StateValidityCheckerFn(partial(DubinsEnvironment.isStateValid, self))
        )

        # create RRT planner
        self.ss.setPlanner(og.RRTConnect(self.ss.getSpaceInformation()))

        # perform setup steps
        space.setup()
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(
            1.0 / space.getMaximumExtent()
        )

    def EstimateRiskMetric(self, state, trajectory, distance, branch_fact, depth, step_size, policy='random'):
        '''Sampling-based risk metric estimation at specific state
        
        Args:
            state : ob.State
                state at which to evaluate risk metric
            trajectory : DubinsTrajectorySegment
                trajectory arriving at state 
            distance : double
                state-space-specific distance to sample within
            branch_fact : int
                number of samples to draw
            depth : int
                number of recursive steps to estimate risk
            step_size : float
                step size used to discretize trajectory
            policy : str
                string description of policy to use

        Returns:
            risk_est : float
                coherent risk metric estimate at state
        '''

        # check if state and trajectory are in collision
        z = self.isFailure(state, trajectory, step_size=step_size)

        if z or depth <= 0:
            # state is failure or at leaf of tree
            return float(z)

        # sample reachable states
        samples = self.sampleReachable(state=state, distance=distance, n_samples=branch_fact, policy=policy)

        # recursively compute risk estimates at sampled states
        risk_vals = branch_fact*[None]
        for i in range(branch_fact):
            risk_vals[i] = self.EstimateRiskMetric(
                state=samples[i].beta,
                trajectory=samples[i],
                distance=distance,
                branch_fact=branch_fact,
                depth=depth-1,
                step_size=step_size,
                policy=policy
            )

        # TODO: generalize this so we can use other coherent risk metrics like CVaR
        return np.mean(risk_vals)

        

    def sampleReachable(self, state, distance, n_samples, policy='random'):
        '''Draw n samples from state space near a given state using a policy

        Args:
            state : ob.State
                state for which nearby samples are to be drawn
            distance : double
                state-space-specific distance to sample within
            n_samples : int
                number of samples to draw
            policy : str
                string description of policy to use

        Returns:
            traj_samples : list(DubinsTrajectorySegments)
                list of dubins trajectory segments 
            # samples : list(ob.State)
            #     list of sampled states
            # paths : list(dubins._DubinsPath)
            #     list of dubins paths to sampled states
        '''

        if policy == 'random':
            sampler = self.ss.getStateSpace().allocDefaultStateSampler()
            # samples = []
            # paths = []
            traj_samples = []
            for i in range(n_samples):

                # sample nearby random state
                s = ob.State(self.ss.getStateSpace())
                sampler.sampleUniformNear(s(), state(), distance)
                # samples.append(s)

                # compute dubins path to sampled state
                p = dubins.shortest_path(
                    q0 = (state().getX(), state().getY(), state().getYaw()),
                    q1 = (s().getX(), s().getY(), s().getYaw()),
                    rho = self.rho
                )
                # paths.append(p)

                traj_samples.append(DubinsTrajectorySegment(alpha=state, beta=s, path=p))
        else:
            raise NotImplementedError("No reachable set sampling implemented for policy {}".format(policy))

        return traj_samples

    def plan(self, 
        start_x, start_y, start_yaw, 
        goal_x, goal_y, goal_yaw,
        plan_iter):
        if not self.ss:
            return False

        # encode start and goal state
        start = ob.State(self.ss.getStateSpace())
        start().setX(start_x)
        start().setY(start_y)
        start().setYaw(start_yaw) 
        goal = ob.State(self.ss.getStateSpace())
        goal().setX(goal_x)
        goal().setY(goal_y)
        goal().setYaw(goal_yaw) 
        self.ss.setStartAndGoalStates(start, goal)

        # generate a few solutions, all will be added to the goal
        for _ in range(plan_iter):
            if self.ss.getPlanner():
                self.ss.getPlanner().clear()
            self.ss.solve()
        ns = self.ss.getProblemDefinition().getSolutionCount()
        print("Found %d solution(s)" % ns)
        if self.ss.haveSolutionPath():
            self.ss.simplifySolution()
            p = self.ss.getSolutionPath()
            ps = og.PathSimplifier(self.ss.getSpaceInformation())
            ps.simplifyMax(p)
            ps.smoothBSpline(p)
            return True
        return False

    def isFailure(self, state, trajectory, step_size):
        ''' check if state and arriving trajectory are in collision with obstacles

        Args:
            state : ob.State
                state to check for failure
            trajectory : DubinsTrajectorySegment
                trajectory leading to state
            step_size : float
                step size used to discretize trajectory
        
        Returns
            fail : bool
                True if state or trajectory are in collision with obs
        '''

        is_state_valid = self.isStateValid(state=state())

        if not is_state_valid:
            return True
        elif trajectory is not None:
            disc_traj = trajectory.discrete_path(step_size)

            # check each discretized state, return if any are in collision
            for d in disc_traj:
                # NOTE: this could be parallelized
                if self.isStateValid(state=d()):
                    pass
                else:
                    return True

            # no invalid states found in trajectory
            return False
            
        else:
            return False

    def isStateValid(self, state):
        ''' check ppm image colors for obstacle collision
        '''
        w = min(int(state.getX()), self.maxWidth)
        h = min(int(state.getY()), self.maxHeight)

        if w < 0 or h < 0 or w > self.maxWidth or h > self.maxHeight:
            return False
        
        c = self.ppm.getPixel(h, w)
        # print(w, h, c)
        tr = c.red > 127
        tg = c.green > 127
        tb = c.green > 127
        return tr and tg and tb

    def recordSolution(self):
        if not self.ss or not self.ss.haveSolutionPath():
            return
        p = self.ss.getSolutionPath()
        p.interpolate()
        self.ppm_output = ou.PPM()
        self.ppm_output.loadFile(self.ppm_file)
        for i in range(p.getStateCount()):
            w = min(self.maxWidth, int(p.getState(i).getX()))
            h = min(self.maxHeight, int(p.getState(i).getY()))
            c = self.ppm_output.getPixel(h, w)
            c.red = 255
            c.green = 0
            c.blue = 0
  
    def save(self, filename):
        if not self.ss:
            return
        self.ppm_output.saveFile(filename)

 
if __name__ == "__main__":
    
    # initial state and environment turning radius in pixels
    x0 = 500
    y0 = 100
    theta0 = 0
    rho = 50
    sample_distance = 100
    path_step_size = 10
    n_samples = 5

    # create planning environment from image
    fname = 'border_640x400.ppm'
    fpath = str(pathlib.Path(__file__).parent.resolve().joinpath(fname))
    env = DubinsEnvironment(fpath, rho)

    # try finding an obstacle-free plan from one state to another
    res_file = "result_demo.ppm"
    if env.plan(
        start_x=x0,
        start_y=y0,
        start_yaw=theta0,
        goal_x=100,
        goal_y=300,
        goal_yaw=0,
        plan_iter=5):
        print("\nFound a Solution!\nRecording to {}\n".format(res_file))
        env.recordSolution()
        env.save(res_file)
    else:
        print("\nNo Solution Found!\n")

    # generate reachable set samples
    s0 = ob.State(env.ss.getStateSpace())
    s0().setX(x0)
    s0().setY(y0)
    s0().setYaw(theta0)
    trajs = env.sampleReachable(s0, sample_distance, n_samples)
    X = [s.beta().getX() for s in trajs]
    Y = [s.beta().getY() for s in trajs]
    TH = [s.beta().getYaw() for s in trajs]
    # plt.arrow(s0().getX(), s0().getY(), 10*np.cos(s0().getYaw()), 10*np.sin(s0().getYaw()),color='g')
    # plt.plot(s0().getX(), s0().getY(), 'gX')
    plt.quiver([s0().getX()], [s0().getY()], [np.cos(s0().getYaw())], [np.sin(s0().getYaw())], color='g')
    # plt.scatter(X, Y)
    plt.quiver(X, Y, np.cos(TH), np.sin(TH), color='b')
    for traj in trajs:
        # discretize and plot the dubins path to a sample
        path_steps, _ = traj.path.sample_many(path_step_size)
        XP = [s[0] for s in path_steps]
        YP = [s[1] for s in path_steps]
        TP = [s[2] for s in path_steps]
        plt.quiver(XP, YP, np.cos(TP), np.sin(TP), scale=50, alpha=0.2)
    plt.show()

    # compute risk metric at s0
    rho_s = env.EstimateRiskMetric(
        state=s0, 
        trajectory=None,
        distance=sample_distance,
        branch_fact=n_samples,
        step_size=path_step_size,
        depth=5)
    print("Estimated risk metric at {}: {}".format(s0, rho_s))

        
    print("Done!")
