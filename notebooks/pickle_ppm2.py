import pickle
import copyreg
from ompl import base as ob
from ompl import util as ou

# Wrap ou.PPM in a pickleable class
class PickleablePPM(ou.PPM):
    def __init__(self, load_file):
        super().__init__()
        self.load_file = load_file
        self.loadFile(self.load_file)

    def __reduce__(self):
        return (PickleablePPM, (self.load_file,))

# ensure that PPM object can now be pickled
ppm = PickleablePPM('floor.ppm')
assert pickle.dumps(ppm)

# ensure that you can copy and reproduce an exact PPM object
ppm_copy = pickle.loads(pickle.dumps(ppm))
print(len(ppm_copy.getPixels()))
assert ppm.getWidth() == ppm_copy.getWidth()
assert ppm.getHeight() == ppm_copy.getHeight()
assert len(ppm.getPixels()) == len(ppm_copy.getPixels())
for i, pix in enumerate(ppm.getPixels()):
    pix.red == ppm_copy.getPixels()[i].red
    pix.green == ppm_copy.getPixels()[i].green
    pix.blue == ppm_copy.getPixels()[i].blue
