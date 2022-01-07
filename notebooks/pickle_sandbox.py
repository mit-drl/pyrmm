import dill
import pickle
import copyreg
import multiprocess
import ompl

from pathlib import Path
from joblib import Parallel, delayed
from types import MethodType
from ompl import util as ou
from ompl import base as ob

from pyrmm.setups.dubins import DubinsPPMSetup

######################################################################
# Approach: https://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-multiprocessing-pool-map/7309686#7309686

# this approach might work if we want a generalized approach for pickling instance methods.
# but that's not what I'm looking for. 
# I want to a way to pickle Boost.Python.class objects. 

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copyreg.pickle(MethodType, _pickle_method, _unpickle_method)
######################################################################

def _pickle_PPM(ppm):
    width = ppm.getWidth()
    height = ppm.getHeight()
    pixels = ppm.getPixels()
    return _unpickle_PPM, (width, height, pixels)

def _unpickle_PPM(width, height, pixels):
    ppm = ou.PPM()
    ppm.setWidth(width)
    ppm.setHeight(height)
    pix = ppm.getPixels()
    pix = pixels
    return ppm

# copyreg.pickle(ou.PPM, _pickle_PPM, _unpickle_PPM)
######################################################################

class PickablePPM(ou.PPM):
    # Ref: https://www.boost.org/doc/libs/1_46_1/libs/python/doc/v2/pickle.html
    # __getstate_manages_dict__ = 1
    def __init__(self):
        super().__init__()

    def __getstate__(self):
        state = dict()
        state['width'] = self.getWidth()
        state['height'] = self.getHeight()
        state['pixels'] = self.getPixels()
        return state

    def __setstate__(self, state):
        self.setWidth(state['width'])
        self.setHeight(state['height'])
        pix = self.getPixels()
        pix = state['pixels']

class PickableDummy(object):
    pass

def create_ppm(ppm_file):
    ppm = ou.PPM()
    ppm.loadFile(ppm_file)
    # return ppm_obj
    return ppm.getWidth(), ppm.getHeight()

# attempt to pickle the object
# pickle.dumps(ppm_obj)

num_cores = multiprocess.cpu_count()
ppm_pth = str(Path(__file__).parent.resolve().joinpath('border_640x400.ppm'))
print("pickled ppm path string: ", pickle.dumps(ppm_pth))
print("pickled ppm creation function: ", pickle.dumps(create_ppm))
print("pickled ppm creation output: ", pickle.dumps(create_ppm(ppm_pth)))

# multiprocess implementation
# partial_estimateRiskMetric = partial(dubss.estimateRiskMetric, 
#     trajectory=None,
#     distance=DURATION,
#     branch_fact=N_BRANCHS[0],
#     depth=DEPTHS[0],
#     n_steps=N_STEPS
# )
# with multiprocess.Pool(num_cores) as pool:
#     rmetrics = pool.map(partial_estimateRiskMetric, ssamples)

# joblib implementation
info = Parallel(n_jobs=num_cores)(
    delayed(create_ppm)(ppm_pth) for _ in range(100)
)

print("\noutput of parallelized PPM creation: ", info)

se2space = ob.SE2StateSpace()

def _pickle_SE2State(state):
    x = state().getX()
    y = state().getY()
    yaw = state().getYaw()
    return _unpickle_SE2State, (x, y, yaw)

def _unpickle_SE2State(x, y, yaw):
    state = ob.SE2State(se2space)
    state().setX(x)
    state().setY(y)
    state().setYaw(yaw)
    return state

copyreg.pickle(ob.SE2State, _pickle_SE2State, _unpickle_SE2State)

def _pickle_SE2StateInternal(state):
    x = state.getX()
    y = state.getY()
    yaw = state.getYaw()
    return _unpickle_SE2StateInternal, (x, y, yaw)

def _unpickle_SE2StateInternal(x, y, yaw):
    state = se2space.allocState()
    state.setX(x)
    state.setY(y)
    state.setYaw(yaw)
    return state

copyreg.pickle(se2space.SE2StateInternal, _pickle_SE2StateInternal, _unpickle_SE2StateInternal)

# create a SE2 state
# s = sspace.allocState()
s = ob.SE2State(se2space)
s().setX(0)
s().setY(0)
s().setYaw(0)
print("\npickled SE2State: ", pickle.dumps(s))

# create and pickle SE2StateInternal
sintern = se2space.allocState()
sintern.setX(10)
sintern.setY(10)
sintern.setYaw(10)
print("\npickled SE2StateInternal: ", pickle.dumps(sintern))
sintern_copy = pickle.loads(pickle.dumps(sintern))
assert se2space.equalStates(sintern, sintern_copy)

# create a dubins setup
dubss = DubinsPPMSetup(ppm_pth, 1, 1)
print("\npickled DubinsPPMSetup", pickle.dumps(dubss))
dubss2 = pickle.loads(pickle.dumps(dubss))
# valids = Parallel(n_jobs=num_cores)(
#     delayed(dubss.dummyFunc)(i) for i in range(100)
# )
# print("result of parallel validity check", valids)

print("\nDone!\n")