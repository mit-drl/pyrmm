'''
a simple example using ompl's dubins vehicle

Ref: 
+ https://ompl.kavrakilab.org/python.html
+ https://ompl.kavrakilab.org/RigidBodyPlanningWithODESolverAndControls_8py_source.html
+ https://ompl.kavrakilab.org/Point2DPlanning_8py_source.html
'''

import pathlib

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from functools import partial


 
def isStateValid(state):
    # "state" is of type SE2StateInternal, so we don't need to use the "()"
    # operator.
    #
    # Some arbitrary condition on the state (note that thanks to
    # dynamic type checking we can just call getX() and do not need
    # to convert state to an SE2State.)
    return state.getX() < .51
 
def plan():
    # create an SE2 state space
    space = ob.DubinsStateSpace()
 
    # set lower and upper bounds
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-1)
    bounds.setHigh(1)
    space.setBounds(bounds)
 
    # create a simple setup object
    ss = og.SimpleSetup(space)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn(isStateValid))
 
    start = ob.State(space)
    # we can pick a random start state...
    start.random()
    # ... or set specific values
    start().setX(.5)
    start().setY(0)
    start().setYaw(0)
 
    goal = ob.State(space)
    # we can pick a random goal state...
    goal.random()
    # ... or set specific values
    goal().setX(.4)
    goal().setY(0)
    goal().setYaw(0)
 
    ss.setStartAndGoalStates(start, goal)
 
    # this will automatically choose a default planner with
    # default parameters
    solved = ss.solve(1.0)
 
    if solved:
        # try to shorten the path
        ss.simplifySolution()
        # print the simplified path
        print(ss.getSolutionPath())
 

class DubinsEnvironment:
    def __init__(self, ppm_file):
        '''
        Args:
            ppm_file: str
                file path to ppm image used as environment map
        '''
        self.ppm = ou.PPM()
        self.ppm.loadFile(ppm_file)
        space = ob.DubinsStateSpace()
        # space.addDimension(0.0, self.ppm.getWidth())
        # space.addDimension(0.0, self.ppm.getHeight())
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
        space.setup()
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(
            1.0 / space.getMaximumExtent()
        )

    def plan(self, start_row, start_col, start_yaw, goal_row, goal_col, goal_yaw):
        if not self.ss:
            return False

        # encode start and goal state
        start = ob.State(self.ss.getStateSpace())
        # start()[0] = start_row
        # start()[1] = start_col
        start().setX(start_row)
        start().setY(start_col)
        start().setYaw(start_yaw) 
        goal = ob.State(self.ss.getStateSpace())
        # goal()[0] = goal_row
        # goal()[1] = goal_col
        goal().setX(goal_row)
        goal().setY(goal_col)
        goal().setYaw(goal_yaw) 
        self.ss.setStartAndGoalStates(start, goal)

        # generate a few solutions, all will be added to the goal
        for _ in range(10):
            if self.ss.getPlanner():
                self.ss.getPlanner().clear()
            self.ss.solve()
        ns = self.ss.getProblemDefinition().getSolutionCount()
        print("Found %d solution" % ns)
        if self.ss.haveSolutionPath():
            self.ss.simplifySolution()
            p = self.ss.getSolutionPath()
            ps = og.PathSimplifier(self.ss.getSpaceInformation())
            ps.simplifyMax(p)
            ps.smoothBSpline(p)
            return True
        return False

    def isStateValid(self, state):
        ''' check ppm image colors for obstacle collision
        '''
        w = min(int(state.getY()), self.maxWidth)
        h = min(int(state.getX()), self.maxHeight)

        if w < 0 or h < 0:
            return False
        
        c = self.ppm.getPixel(h, w)
        # print(w, h, c)
        tr = c.red > 127
        tg = c.green > 127
        tb = c.green > 127
        return tr and tg and tb

 
if __name__ == "__main__":
    
    # create planning environment from image
    fname = 'border_640x400.ppm'
    fpath = str(pathlib.Path(__file__).parent.resolve().joinpath(fname))
    env = DubinsEnvironment(fpath)

    env.plan(0,0,0,10,10,0)
    print("Done!")
