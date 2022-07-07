"""
Functions for defining and using Quadrotor dynamics

Ref: https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

# Variable Notation:
# v__x: vector expressed in "x" frame
# q_x_y: quaternion of "x" frame with respect to "y" frame
# p_x_y__z: position of "x" frame with respect to "y" frame expressed in "z" coordinates
# v_x_y__z: velocity of "x" frame with respect to "y" frame expressed in "z" coordinates
# R_x2y: rotation matrix that maps vector represented in frame "x" to representation in frame "y" (right-multiply column vec)
#
# Frame Subscripts:
# dc = downward-facing camera (body-fixed, non-inertial frame. Origin: downward camera focal plane. Alignment with respect to drone airframe: x-forward, y-right, z-down)
# fc = forward-facing camera (body-fixed, non-inertial frame. Origin: forward camera focal plane. Alignment with respect to drone airframe: x-right, y-down, z-forward)
# bu = body-up frame (body-fixed, non-inertial frame. Origin: drone center of mass. Alignment with respect to drone airframe: x-forward, y-left, z-up)
# bd = body-down frame (body-fixed, non-inertial frame. Origin: drone center of mass. Alignment with respect to drone airframe: x-forward, y-right, z-down)
# wu = world-up frame (world-fixed, inertial frame. Origin: arbitrarily defined. Alignment with respect to world: x-(arbitrary), y-(arbitrary), z-up)
# wd = world-down frame (body-fixed, non-inertial frame. Origin: drone center of mass. Alignment with respect to world: x-(arbitrary), y-(arbitrary), z-down)
# lenu = local East-North-Up world frame (world-fixed, inertial frame. Origin: apprx at take-off point, but not guaranteed. Alignment with respect to world: x-East, y-North, z-up)
# lned = local North-East-Down world frame (world-fixed, inertial frame. Origin: apprx at take-off point, but not guaranteed. Alignment with respect to world: x-North, y-East, z-down)
# m = marker frame (inertial or non-inertial, depending on motion of marker. Origin: center of marker. Alignment when looking at marker: x-right, y-up, z-out of plane toward you)

"""
import numpy as np
import pybullet as pb
from ompl import base as ob
from ompl import control as oc

class QuadrotorStateSpace(ob.CompoundStateSpace):
    '''6DoF state space defined as OMPL CompoundStateSpace'''
    def __init__(self, bounds=None):
        '''
        Args
            bounds : Dict
                dictionary of state space bounds. 
                If None, no bounds set (but this causes sampler to sample trivial states)
                If not None, then all entries must be given
                pos_low : list-like
                    minimum position states [m]
                pos_high : list-like
                    maximum position states [m]
                vel_low : list-like
                    minimum velocity states [m/s]
                vel_high : list-like
                    maximum velocity states [m/s]
                omg_low : list-like
                    minimum angular velocity states [rad/s]
                omg_high : list-like
                    maximum angular velocity states [rad/s]

        '''
        super().__init__()
        self.addSubspace(ob.RealVectorStateSpace(3), 1.0)   # position
        self.addSubspace(ob.SO3StateSpace(), 1.0)           # orientation
        self.addSubspace(ob.RealVectorStateSpace(3), 1.0)   # velocity
        self.addSubspace(ob.RealVectorStateSpace(3), 1.0)   # angular velocity

        # set bounds
        if bounds is not None:

            # ensure bounds properly configured
            req_keys = ['pos_low', 'pos_high', 'vel_low', 'vel_high', 'omg_low', 'omg_high']
            assert all([k in bounds.keys() for k in req_keys])

            # set position bounds
            assert all(np.greater_equal(bounds['pos_high'], bounds['pos_low']))
            pos_bounds = ob.RealVectorBounds(3)
            pos_bounds.setLow(0, bounds['pos_low'][0]); pos_bounds.setHigh(0, bounds['pos_high'][0])
            pos_bounds.setLow(1, bounds['pos_low'][1]); pos_bounds.setHigh(1, bounds['pos_high'][1])
            pos_bounds.setLow(2, bounds['pos_low'][2]); pos_bounds.setHigh(2, bounds['pos_high'][2])
            self.getSubspace(0).setBounds(pos_bounds)

            # set velocity bounds
            assert all(np.greater_equal(bounds['vel_high'], bounds['vel_low']))
            vel_bounds = ob.RealVectorBounds(3)
            vel_bounds.setLow(0, bounds['vel_low'][0]); vel_bounds.setHigh(0, bounds['vel_high'][0])
            vel_bounds.setLow(1, bounds['vel_low'][1]); vel_bounds.setHigh(1, bounds['vel_high'][1])
            vel_bounds.setLow(2, bounds['vel_low'][2]); vel_bounds.setHigh(2, bounds['vel_high'][2])
            self.getSubspace(2).setBounds(vel_bounds)

            # set angular velocity bounds
            assert all(np.greater_equal(bounds['omg_high'], bounds['omg_low']))
            omg_bounds = ob.RealVectorBounds(3)
            omg_bounds.setLow(0, bounds['omg_low'][0]); omg_bounds.setHigh(0, bounds['omg_high'][0])
            omg_bounds.setLow(1, bounds['omg_low'][1]); omg_bounds.setHigh(1, bounds['omg_high'][1])
            omg_bounds.setLow(2, bounds['omg_low'][2]); omg_bounds.setHigh(2, bounds['omg_high'][2])
            self.getSubspace(3).setBounds(omg_bounds)

class QuadrotorThrustMomentControlSpace(oc.RealVectorControlSpace):
    '''Quadrotor control via body z-axis thrust and moments'''
    def __init__(self, stateSpace, fzmax, mxmax, mymax, mzmax):
        '''
        Args:
            stateSpace : ob.StateSpace
                state space associated with quadrotor (pos, orn, vel, ang vel)
            fzmax : float
                max net force from all rotors along body z axis (N)
            mxmax : float
                max net moment applied by rotors along body x axis (N*m)
            mymax : float
                max net moment applied by rotors along body y axis (N*m)
            mzmax : float
                max net moment applied by rotors along body z axis (N*m)
        '''
        assert fzmax > 0
        assert mxmax > 0
        assert mymax > 0
        assert mzmax > 0
        super().__init__(stateSpace=stateSpace, dim=4)

        # set control bounds: fz always positive, moments symmetric positiv and negative
        cbounds = ob.RealVectorBounds(4)
        cbounds.setLow(0, 0); cbounds.setHigh(0, fzmax)
        cbounds.setLow(1, -mxmax); cbounds.setHigh(1, mxmax)
        cbounds.setLow(2, -mymax); cbounds.setHigh(2, mymax)
        cbounds.setLow(3, -mzmax); cbounds.setHigh(3, mzmax)
        self.setBounds(cbounds)

def body_thrust_torque_physics(control, pbQuadId, pbClientId):
        '''physics model based on body-fixed thrust (z-axis aligned force) and torque controls
        Args:
            controls : oc.ControlType
                OMPL ControlType from QuadrotorThrustMomentControlSpace defining
                force and torques in body axes: [F_zb, M_xb, M_yb, M_zb]
            pbBodyId : int
                PyBullet unique object ID of quadrotor body associated with propagator
            pbClientId : int
                ID number of pybullet physics client

        Ref:
            Allen, "A real-time framework for kinodynamic planning in dynamic environments with application to quadrotor obstacle avoidance",
            Sec 4.1
        '''
        # extract and apply body z-axis aligned thrus
        thrust = control[0]
        pb.applyExternalForce(objectUniqueId=pbQuadId,
                                linkIndex=-1,
                                forceObj=[0, 0, thrust],
                                posObj=[0, 0, 0],
                                flags=pb.LINK_FRAME,
                                physicsClientId=pbClientId
                                )

        # extract and apply body torques
        moments = [control[1], control[2], control[3]]
        pb.applyExternalTorque(objectUniqueId=pbQuadId,
                              linkIndex=-1,
                              torqueObj=moments,
                              flags=pb.LINK_FRAME,
                              physicsClientId=pbClientId
                              )
