'''Convert stl mesh files to obj using blender

Refs:
 + https://stackoverflow.com/questions/38096913/blender-convert-stl-to-obj-with-prompt-commande
 + https://blender.stackexchange.com/questions/27234/python-how-to-completely-remove-an-object
'''

import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:] # get all args after --

stl_in = argv[0]
obj_out = argv[1]

# remove cube object
objs = bpy.data.objects
objs.remove(objs["Cube"], do_unlink=True)

bpy.ops.import_mesh.stl(filepath=stl_in, axis_forward='-Z', axis_up='Y')
bpy.ops.export_scene.obj(filepath=obj_out, axis_forward='-Z', axis_up='Y')

