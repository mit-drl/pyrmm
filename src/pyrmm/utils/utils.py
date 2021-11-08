'''
Utility functions and classes for risk metric maps

Examples: collision checkers, tree and node definitions
'''

from typing import List

class Node2D:
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = [x]
        self.path_y = [y]
        self.parent = None

def check_collision_2D_circular_obstacles(node: Node2D, obstacleList: List):
    ''' check if node, and path to node, is in collision with any circular obstacles

    Args:
        node: Node
            state node to check for collisions, including path to the obstacle
        obstacleList: List
            list of circular obstacles, each formated as (x-pos, y-pos, radius)
        
    Returns:
        is_collision: bool
            True if node is in collision with any obstacle, otherwise false
    '''

    if node is None:
        return True

    for (ox, oy, size) in obstacleList:
        dx_list = [ox - x for x in node.path_x]
        dy_list = [oy - y for y in node.path_y]
        d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

        if min(d_list) <= size**2:
            return True  # collision

    return False  # safe