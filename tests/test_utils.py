import pyrmm.utils.utils as U

def test_check_collision_circular_obstacles_simple():
    # check for collision with a single obstacle at origin
    a = U.Node2D(0,0)
    o = [(0, 0, 1)]
    assert U.check_collision_2D_circular_obstacles(a,o)