# Procedurally generates a set of mazes using https://github.com/razimantv/mazegenerator

import subprocess
import pyvips

# generate mazes
mazegen_cmd = '../../../mazegenerator/src/mazegen -o my_maze'
mazegen_proc = subprocess.Popen(mazegen_cmd.split(), stdout=subprocess.PIPE)
output, error = mazegen_proc.communicate()

# convert to ppm
image = pyvips.Image.new_from_file('my_maze.svg', dpi=300)
image.write_to_file('my_maze.ppm')
