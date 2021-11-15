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
 

class DubinsEnvironment:
    def __init__(self, ppm_file, turn_rad):
        '''
        Args:
            ppm_file: str
                file path to ppm image used as environment map
            turn_rad: float
                turning radius of the dubins vehicle
        '''
        self.ppm = ou.PPM()
        self.ppm.loadFile(ppm_file)
        space = ob.DubinsStateSpace(turningRadius=turn_rad)
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

        # create RRT planner
        self.ss.setPlanner(og.RRTConnect(self.ss.getSpaceInformation()))

        # perform setup steps
        space.setup()
        self.ss.getSpaceInformation().setStateValidityCheckingResolution(
            1.0 / space.getMaximumExtent()
        )

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
        for i in range(p.getStateCount()):
            w = min(self.maxWidth, int(p.getState(i).getX()))
            h = min(self.maxHeight, int(p.getState(i).getY()))
            c = self.ppm.getPixel(h, w)
            c.red = 255
            c.green = 0
            c.blue = 0
  
    def save(self, filename):
        if not self.ss:
            return
        self.ppm.saveFile(filename)

 
if __name__ == "__main__":
    
    # create planning environment from image
    fname = 'border_640x400.ppm'
    fpath = str(pathlib.Path(__file__).parent.resolve().joinpath(fname))
    turn_rad = 50
    env = DubinsEnvironment(fpath, turn_rad)

    res_file = "result_demo.ppm"
    if env.plan(
        start_x=500,
        start_y=100,
        start_yaw=0,
        goal_x=100,
        goal_y=300,
        goal_yaw=0,
        plan_iter=5):
        print("\nFound a Solution!\nRecording to {}\n".format(res_file))
        env.recordSolution()
        env.save(res_file)
    else:
        print("\nNo Solution Found!\n")
        
    print("Done!")
