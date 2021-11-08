'''
a simple example using ompl's dubins vehicle

Ref: 
+ https://ompl.kavrakilab.org/python.html
+ https://ompl.kavrakilab.org/RigidBodyPlanningWithODESolverAndControls_8py_source.html
'''

from ompl import base as ob
from ompl import geometric as og
 
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
 
 
if __name__ == "__main__":
    plan()