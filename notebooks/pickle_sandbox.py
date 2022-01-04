import dill as pickle
import copyreg

from pathlib import Path
from types import MethodType
from ompl import util as ou

# create ppm object from ompl utils
ppm_file = 'border_640x400.ppm'
ppm_pth = str(Path(__file__).parent.resolve().joinpath(ppm_file))
ppm_obj = ou.PPM()
ppm_obj.loadFile(ppm_file)

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

# attempt to pickle the object
pickle.dumps(ppm_obj)