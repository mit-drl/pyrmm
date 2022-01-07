import pickle
import copyreg
from ompl import base as ob
from ompl import util as ou

# Wrap ou.PPM in a pickleable class
class PickleablePPM(ou.PPM):
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

    def __reduce__(self):
        return (PickleablePPM, (), self.__getstate__())

# define pickling function for vectorPPMColor used to define pixels in PPM
def pickle_vectorPPMColor(pixels):
    rgb = [(pix.red, pix.green, pix.blue) for pix in pixels]
    return unpickle_vectorPPMColor, (rgb,)

# define un-pickling function for vectorPPMColor
def unpickle_vectorPPMColor(rgb):
    pixels = ou.vectorPPMColor()
    for (r,g,b) in rgb:
        color = ou.PPM.Color()
        color.red = r
        color.green = g
        color.blue = b
        pixels.append(color)
    return pixels

# tell pickle how to pickle vectorPPMColor objects
copyreg.pickle(ou.vectorPPMColor, pickle_vectorPPMColor, unpickle_vectorPPMColor)  

# ensure that PPM object can now be pickled
ppm = PickleablePPM()
assert pickle.dumps(ppm)

# ensure that you can copy and reproduce an exact PPM object
ppm.loadFile('floor.ppm')
ppm_copy = pickle.loads(pickle.dumps(ppm))
assert ppm.getWidth() == ppm_copy.getWidth()
assert ppm.getHeight() == ppm_copy.getHeight()
for i, p1, p2 in enumerate(zip(ppm.getPixels(), ppm_copy.getPixels())):
    assert p1.red == p2.red
    assert p1.green == p2.green
    assert p1.blue == p2.blue
