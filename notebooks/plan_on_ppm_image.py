#!/usr/bin/env python



# Author: Ioan Sucan, Mark Moll
# Ref: https://ompl.kavrakilab.org/Point2DPlanning_8py_source.html

import pathlib
from os.path import abspath, dirname, join
import sys
try:
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    sys.path.insert(0, join(dirname(dirname(abspath(__file__))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og
from functools import partial

class Plane2DEnvironment:
    def __init__(self, ppm_file):
        self.ppm_ppm_ = ou.PPM()
        self.ppm_ppm_.loadFile(ppm_file)
        space = ob.RealVectorStateSpace()
        space.addDimension(0.0, self.ppm_ppm_.getWidth())
        space.addDimension(0.0, self.ppm_ppm_.getHeight())
        self.maxWidth_maxWidth_ = self.ppm_ppm_.getWidth() - 1
        self.maxHeight_maxHeight_ = self.ppm_ppm_.getHeight() - 1
        self.ss_ss_ = og.SimpleSetup(space)

        # set state validity checking for this space
        self.ss_ss_.setStateValidityChecker(ob.StateValidityCheckerFn(
            partial(Plane2DEnvironment.isStateValid, self)))
        space.setup()
        self.ss_ss_.getSpaceInformation().setStateValidityCheckingResolution( \
            1.0 / space.getMaximumExtent())
        #      self.ss_.setPlanner(og.RRTConnect(self.ss_.getSpaceInformation()))

    def plan(self, start_row, start_col, goal_row, goal_col):
        if not self.ss_ss_:
            return False
        start = ob.State(self.ss_ss_.getStateSpace())
        start()[0] = start_row
        start()[1] = start_col
        goal = ob.State(self.ss_ss_.getStateSpace())
        goal()[0] = goal_row
        goal()[1] = goal_col
        self.ss_ss_.setStartAndGoalStates(start, goal)
        # generate a few solutions; all will be added to the goal
        for _ in range(10):
            if self.ss_ss_.getPlanner():
                self.ss_ss_.getPlanner().clear()
            self.ss_ss_.solve()
        ns = self.ss_ss_.getProblemDefinition().getSolutionCount()
        print("Found %d solutions" % ns)
        if self.ss_ss_.haveSolutionPath():
            self.ss_ss_.simplifySolution()
            p = self.ss_ss_.getSolutionPath()
            ps = og.PathSimplifier(self.ss_ss_.getSpaceInformation())
            ps.simplifyMax(p)
            ps.smoothBSpline(p)
            return True
        return False

    def recordSolution(self):
        if not self.ss_ss_ or not self.ss_ss_.haveSolutionPath():
            return
        p = self.ss_ss_.getSolutionPath()
        p.interpolate()
        for i in range(p.getStateCount()):
            w = min(self.maxWidth_maxWidth_, int(p.getState(i)[0]))
            h = min(self.maxHeight_maxHeight_, int(p.getState(i)[1]))
            c = self.ppm_ppm_.getPixel(h, w)
            c.red = 255
            c.green = 0
            c.blue = 0

    def save(self, filename):
        if not self.ss_ss_:
            return
        self.ppm_ppm_.saveFile(filename)

    def isStateValid(self, state):
        w = min(int(state[0]), self.maxWidth_maxWidth_)
        h = min(int(state[1]), self.maxHeight_maxHeight_)

        c = self.ppm_ppm_.getPixel(h, w)
        return c.red > 127 and c.green > 127 and c.blue > 127


if __name__ == "__main__":
    # fname = join(join(join(join(dirname(dirname(abspath(__file__))), \
    #     'tests'), 'resources'), 'ppm'), 'floor.ppm')
    # fname = join(dirname(dirname(abspath(__file__))), 'floor.ppm')
    fname = str(pathlib.Path(__file__).parent.resolve().joinpath('floor.ppm'))
    env = Plane2DEnvironment(fname)

    if env.plan(0, 0, 777, 1265):
        env.recordSolution()
        env.save("result_demo.ppm")
