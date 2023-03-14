'''
SystemSetup for Quadrotor vehicle

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

'''

import copyreg
import pathlib
import numpy as np
import pybullet as pb

from functools import partial

from ompl import base as ob
from ompl import control as oc

from pyrmm.setups import SystemSetup
import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD

# hard-coded default setup parameters
_DEFAULT_QUAD_URDF = str(pathlib.Path(U.get_repo_path()).joinpath("tests/quadrotor.urdf"))
_DEFAULT_POSXMIN = -100.0       # [m]
_DEFAULT_POSXMAX = 100.0        # [m]
_DEFAULT_POSYMIN = -100.0       # [m] 
_DEFAULT_POSYMAX = 100.0        # [m]
_DEFAULT_POSZMIN = 0.0          # [m]
_DEFAULT_POSZMAX = 100.0        # [m]
_DEFAULT_VELXMIN = -10.0        # [m/s]
_DEFAULT_VELXMAX = 10.0         # [m/s]
_DEFAULT_VELYMIN = -10.0        # [m/s] 
_DEFAULT_VELYMAX = 10.0         # [m/s]
_DEFAULT_VELZMIN = -10.0        # [m/s]
_DEFAULT_VELZMAX = 10.0         # [m/s]
_DEFAULT_OMGXMIN = -4*np.pi     # [rad/s]
_DEFAULT_OMGXMAX = 4*np.pi      # [rad/s]
_DEFAULT_OMGYMIN = -4*np.pi     # [rad/s] 
_DEFAULT_OMGYMAX = 4*np.pi      # [rad/s]
_DEFAULT_OMGZMIN = -4*np.pi     # [rad/s]
_DEFAULT_OMGZMAX = 4*np.pi      # [rad/s]
_DEFAULT_FZMAX=8.0              # [N]
_DEFAULT_MXMAX=2.0              # [N*m]
_DEFAULT_MYMAX=2.0              # [N*m]
_DEFAULT_MZMAX=1.0              # [N*m]

class QuadrotorPyBulletSetup(SystemSetup):
    ''' Quadrotor vehicle defined with PyBullet physics and configuration space
    '''
    def __init__(self, lidar_range=None, lidar_angles=None, **kwargs):
        '''
        Args:
            lidar_range : float
                range of each ray to cast [m]
            lidar_angles : list(tuples)
                list of tuples describing angle of each ray [rad] relative to body orientation
                tuples give (polar,azimuth) angles.
                See physics spherical coord convention: https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg
            kwargs : dict (Optional)
                pb_client_id : int
                    pybullet client ID number for existing connection, if it exists
                quad_urdf : str
                    path to quadrotor urdf robot description file
                fzmax : float
                    max net force from all rotors along body z axis (N)
                mxmax : float
                    max net moment applied by rotors along body x axis (N*m)
                mymax : float
                    max net moment applied by rotors along body y axis (N*m)
                mzmax : float
                    max net moment applied by rotors along body z axis (N*m)

        '''

        # TODO: assert physical inputs

        # save init args for re-creation of object
        self.lidar_range = lidar_range
        self.lidar_angles = lidar_angles

        # generate configuration space
        # connect to headless physics engine
        if 'pb_client_id' in kwargs:
            self.pbClientId = kwargs['pb_client_id']
        else:
            self.pbClientId = pb.connect(pb.DIRECT)
        assert pb.isConnected(self.pbClientId)

        # create pybullet instance of quadrotor
        self.pbBodyId = pb.loadURDF(_DEFAULT_QUAD_URDF)

        # create compound state space (pos, quat, vel, ang_vel) and set bounds
        pxmin = _DEFAULT_POSXMIN if 'pxmin' not in kwargs else kwargs['pxmin']
        pxmax = _DEFAULT_POSXMAX if 'pxmax' not in kwargs else kwargs['pxmax']
        pymin = _DEFAULT_POSYMIN if 'pymin' not in kwargs else kwargs['pymin']
        pymax = _DEFAULT_POSYMAX if 'pymax' not in kwargs else kwargs['pymax']
        pzmin = _DEFAULT_POSZMIN if 'pzmin' not in kwargs else kwargs['pzmin']
        pzmax = _DEFAULT_POSZMAX if 'pzmax' not in kwargs else kwargs['pzmax']
        vxmin = _DEFAULT_VELXMIN if 'vxmin' not in kwargs else kwargs['vxmin']
        vxmax = _DEFAULT_VELXMAX if 'vxmax' not in kwargs else kwargs['vxmax']
        vymin = _DEFAULT_VELYMIN if 'vymin' not in kwargs else kwargs['vymin']
        vymax = _DEFAULT_VELYMAX if 'vymax' not in kwargs else kwargs['vymax']
        vzmin = _DEFAULT_VELZMIN if 'vzmin' not in kwargs else kwargs['vzmin']
        vzmax = _DEFAULT_VELZMAX if 'vzmax' not in kwargs else kwargs['vzmax']
        oxmin = _DEFAULT_OMGXMIN if 'oxmin' not in kwargs else kwargs['oxmin']
        oxmax = _DEFAULT_OMGXMAX if 'oxmax' not in kwargs else kwargs['oxmax']
        oymin = _DEFAULT_OMGYMIN if 'oymin' not in kwargs else kwargs['oymin']
        oymax = _DEFAULT_OMGYMAX if 'oymax' not in kwargs else kwargs['oymax']
        ozmin = _DEFAULT_OMGZMIN if 'ozmin' not in kwargs else kwargs['ozmin']
        ozmax = _DEFAULT_OMGZMAX if 'ozmax' not in kwargs else kwargs['ozmax']
        sbounds = dict()
        sbounds['pos_low']= [pxmin, pymin, pzmin]
        sbounds['pos_high'] = [pxmax, pymax, pzmax]
        sbounds['vel_low']= [vxmin, vymin, vzmin]
        sbounds['vel_high'] = [vxmax, vymax, vzmax]
        sbounds['omg_low']= [oxmin, oymin, ozmin]
        sbounds['omg_high'] = [oxmax, oymax, ozmax]
        state_space = QD.QuadrotorStateSpace(bounds=sbounds)

        # create control space, using default set bounds if none given in kwargs
        fzmax = _DEFAULT_FZMAX if 'fzmax' not in kwargs else kwargs['fzmax']
        mxmax = _DEFAULT_MXMAX if 'mxmax' not in kwargs else kwargs['mxmax']
        mymax = _DEFAULT_MYMAX if 'mymax' not in kwargs else kwargs['mymax']
        mzmax = _DEFAULT_MZMAX if 'mzmax' not in kwargs else kwargs['mzmax']
        control_space = QD.QuadrotorThrustMomentControlSpace(
            stateSpace=state_space,
            fzmax=fzmax,
            mxmax=mxmax,
            mymax=mymax,
            mzmax=mzmax)

        # create space information for state and control space
        space_info = oc.SpaceInformation(stateSpace=state_space, controlSpace=control_space)
        self.space_info = space_info

        # create and set state validity checker
        validityChecker = ob.StateValidityCheckerFn(partial(self.isStateValid, space_info))
        space_info.setStateValidityChecker(validityChecker)

        # set gravity assuming world-up reference frame (positive z-axis points anti-gravity)
        pb.setGravity(0, 0, -U.GRAV_CONST, physicsClientId=self.pbClientId)

        # call parent init to create simple setup
        # equations of motion are None because custom 
        # propagate_path function implemented
        super().__init__(
            space_information = space_info,
            eom_ode = None
        )

    def __reduce__(self):
        ''' Function to enable re-creation of unpickable object

        Note: See comments about potential risks here
        https://stackoverflow.com/a/50308545/4055705
        '''
        return (QuadrotorPyBulletSetup, (self.lidar_range, self.lidar_angles))
    
    def isStateValid(self, spaceInformation, state):
        ''' check for collisions using pybullet getContactPoints
        Args:
            spaceInformation : ob.SpaceInformationPtr
                state space information as given by SimpleSetup.getSpaceInformation
            state : ob.State
                state to check for validity
        
        Returns:
            True if state in bound and not in collision with obstacles
        '''

        # reset the pb body state based on propogation start state
        copy_state_ompl2pb(
            pbBodyId=self.pbBodyId, 
            pbClientId=self.pbClientId, 
            omplState=state
        )

        # call collision detection function to assess collisions 
        # independent of simulation stepping
        # ref: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
        pb.performCollisionDetection()

        # get all contacts between quadrotor and other objects in env
        contacts = pb.getContactPoints(
            bodyA = self.pbBodyId,
            physicsClientId = self.pbClientId
        )

        if len(contacts) > 0:
            return False
        else:
            return True

    def observeLidar(self, state, ray_range, ray_angles):
        ''' get simulated lidar data from ray casting at a given state
        Args:
            state : QD.QuadrotorState or None
                ompl-based state from which to make observation
                if state is None, then assume it has already been set (for efficiency)
            ray_range : float
                range of each ray to cast [m]
            ray_angles : list(tuples)
                list of tuples describing angle of each ray [rad] relative to body orientation
                tuples give (polar,azimuth) angles.
                See physics spherical coord convention: https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg
        Returns:
            observation : list-like
                array giving observation values
        '''

        n_rays = len(ray_angles)

        # update pybullet state with ompl state
        if state is not None:
            copy_state_ompl2pb(
                pbBodyId = self.pbBodyId, 
                pbClientId = self.pbClientId,
                omplState = state)

        # position of ray endpoints relative to quadrotor body, experessed in quad body-up coords
        p_ri_bu__bu = n_rays * [(0.0, 0.0, 0.0)]
        p_rf_bu__bu = [U.spherical_to_cartesian(rho=ray_range, theta=ang[0], phi=ang[1]) for ang in ray_angles]

        # Call rayTestBatch with from and to endpoints
        ray_casts = pb.rayTestBatch(
            rayFromPositions = p_ri_bu__bu,
            rayToPositions = p_rf_bu__bu,
            parentObjectUniqueId = self.pbBodyId,
            parentLinkIndex = -1,
            physicsClientId = self.pbClientId)

        return ray_casts

    def observeState(self, state):
        ''' observer obstacle-relative states: lidar, velocity, orientation quaternion, angular rates
        Args:
            state : QD.QuadrotorState
                ompl-based state from which to make observation

        Returns:
            observation : list-like
                array giving observation values
                [0:n_rays] : lidar range readings [m]
                [n_rays:n_rays+4] : quaternion of body frame relative to world frame [x,y,z,w]
                [n_rays+4:n_rays+7] : velocity of body frame relative to world frame [m/s]
                [n_rays+7:n_rays+10] : body frame rate relative to world frame [rad/s]
        '''

        # update pybullet state with ompl state
        copy_state_ompl2pb(
            pbBodyId = self.pbBodyId, 
            pbClientId = self.pbClientId,
            omplState = state)

        # instantiate observation vector
        n_rays = len(self.lidar_angles)
        obs = n_rays * [None]

        # get lidar observations
        # pass None state because it is already copied to pybullet
        rays = self.observeLidar(state=None, ray_range=self.lidar_range, ray_angles=self.lidar_angles)
        obs[:n_rays] = [self.lidar_range * r[2] for r in rays]

        # get orientation observation
        obs[n_rays:n_rays+4] = [state[1].x, state[1].y, state[1].z, state[1].w]

        # get velocity observation
        obs[n_rays+4:n_rays+7] = [state[2][i] for i in range(3)]

        # get angular rate observation
        obs[n_rays+7:] = [state[3][i] for i in range(3)]

        return obs

    def propagate(self, state, control, duration, result, **kwargs):
        ''' propagate from start based on control, store in state
        Args:
            state : ob.State
                start state of propagation
            control : oc.Control
                control to apply during propagation
            duration : float
                duration of propagation
            result : ob.State
                end state of propagation, modified in place
            ret_true_duration : bool (optional)
                flag to signal true duration should be returned

        Notes:
            By default, propagate does not perform or is used in integration,
            even when defined through an ODESolver; see:
            https://ompl.kavrakilab.org/RigidBodyPlanningWithODESolverAndControls_8py_source.html
            https://ompl.kavrakilab.org/classompl_1_1control_1_1StatePropagator.html#a4bf54becfce458e1e8abfa4a37ae8dff

            This implementation uses pybullet's built-in ode propagator.
            Ref: https://github.com/utiasDSL/gym-pybullet-drones/blob/a4e165bcbeb9133bee4bf920fca3d1a170f7bba7/gym_pybullet_drones/envs/BaseAviary.py#L272

            Due to PyBullet's fixed timestep implementation, the duration of propagation is approximate. There is likely to cause
            an "over propagation" beyond the request duration
        '''

        # check that propagation is not too small of time for PyBullet
        dt = pb.getPhysicsEngineParameters()['fixedTimeStep']
        if dt > duration:
            raise Exception("Duration of propagation is less than PyBullet's fixed timestep")

        # reset the pb body state based on propogation start state
        copy_state_ompl2pb(
            pbBodyId=self.pbBodyId, 
            pbClientId=self.pbClientId, 
            omplState=state
        )

        # clip the control to ensure it is within the control bounds
        bounded_control = U.clip_control(controlSpace=self.space_info.getControlSpace(), control=control)

        # call pybullet's simulator step, performs ode propagation
        t = 0.0
        while t < duration:
            # must reset external forces and torques after each sim step
            # ref: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.mq73m9o2gcpy
            QD.body_thrust_torque_physics(bounded_control, self.pbBodyId, self.pbClientId)
            pb.stepSimulation(physicsClientId=self.pbClientId)
            t += dt

        # Extract state information from pb physics client and store in OMPL result
        copy_state_pb2ompl(
            pbBodyId=self.pbBodyId, 
            pbClientId=self.pbClientId, 
            omplState=result)
        
        # return the true duration if requested
        if 'ret_true_duration' in kwargs and kwargs['ret_true_duration']:
            return t

    def propagate_path(self, state, control, duration, path, **kwargs):
        ''' propagate from start based on control, store final state in result, store path to result
        Args:
            state : ob.State
                start state of propagation
            control : oc.Control
                control to apply during propagation
            duration : float
                duration of propagation
            path : oc.ControlPath
                path from state to result in nsteps. initial state is state, final state is result
            ret_true_duration : bool (optional)
                flag to signal true duration should be returned

        Returns:
            t : float (if ret_true_duration flag)
                true duration propagated 

        Notes:
            This function is similar, but disctinct from 'StatePropagator.propagate', thus its different name to no overload `propagate`. 
            propagate does not store or return the path to get to result
            
            This implementation uses pybullet's built-in ode propagator.
            Ref: https://github.com/utiasDSL/gym-pybullet-drones/blob/a4e165bcbeb9133bee4bf920fca3d1a170f7bba7/gym_pybullet_drones/envs/BaseAviary.py#L272
        '''

        # unpack objects from space information for ease of use
        sspace = self.space_info.getStateSpace()
        cspace = self.space_info.getControlSpace()
        nsteps = path.getStateCount()
        pstates = path.getStates()
        pcontrols = path.getControls()
        ptimes = path.getControlDurations()
        assert len(pstates) == nsteps
        assert len(pcontrols) == len(ptimes) == nsteps-1
        assert nsteps >= 2
        assert duration > 0 and not np.isclose(duration, 0.0)

        # compute requested duration of each substep
        req_sub_dur = duration/(nsteps-1)

        # clip the control to ensure it is within the control bounds
        # and so that it can be stored in path's controls (can't be done in propagate func)
        bounded_control = U.clip_control(controlSpace=cspace, control=control)

        # set initial path state equal to initial state
        sspace.copyState(destination=pstates[0], source=state)

        # Call propagate on each step to get each intermediate point on path
        cum_dur = 0.0
        for i in range(nsteps-1):
            cspace.copyControl(destination=pcontrols[i], source=bounded_control)
            true_sub_dur = self.propagate(
                state=pstates[i],
                control=pcontrols[i],
                duration=req_sub_dur,
                result=pstates[i+1],
                ret_true_duration = True
            )
            ptimes[i] = true_sub_dur
            cum_dur += true_sub_dur

        if 'ret_true_duration' in kwargs and kwargs['ret_true_duration']:
            return cum_dur

def proportional_angular_rate_controller_0(omg, omg_sp, kp):
    '''Compute moment control inputs from angular rates based on proportional control

    Args:
        omg : array-like (len=3)
            current angular rates in body x, y, z axes
        omg_sp : array-like (len=3)
            desired (setpoint) angular rates in body x, y, z axes
        kp : array-like (len=3) 
            proportional gains to convert angular rate error to torque

    Returns:
        M : array-like (len=3)
            control moments in body x, y, z axes

    Ref: 
        + http://docs.px4.io/main/en/flight_stack/controller_diagrams.html#multicopter-angular-rate-controller
        + http://docs.px4.io/main/en/config_mc/pid_tuning_guide_multicopter.html#rate-controller
    '''
    # compute rate error
    omg_err = omg_sp - omg

    # return control moments
    return np.multiply(omg_err, kp)

def proportional_quaternion_controller_0(q, q_sp, kp):
    '''Compute angular rate setpoint from quaternion based on proportional control
    
    Args:
        q : array-like (len=4)
            current orientation quaternion in x, y, z, w format
        q_sp : array-like (len=4)
            desired (setpoint) orientation quaternion in x, y, z, w format
        kp : float
            proportional gain to convert quaternion error to angular rate setpoints

    Returns:
        omg_sp : array-like (len=3)
            desired (setpoint) angular rates in body x, y, z axes

    Ref:
        + http://docs.px4.io/main/en/flight_stack/controller_diagrams.html#multicopter-attitude-controller
        + https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf
    '''

    # compute quaternion error
    q_err = pb.getDifferenceQuaternion(q, q_sp)

    # return angural rate setpoint
    return 2*kp * np.sign(q_err[3]) * np.array(q_err[0:3])

def ompl_to_numpy(omplState):
    '''Represent a OMPL-defined quadrotor state as an numpy array
    Args:
        omplState : ob.CompoundState
            ompl's state object for 6DoF with 
            ordering (position, orientation, velocity, angular velocity)
    
    Returns:
        state_np : np.ndarray
            numpy array representation of ompl state with 
            ordering (position, orientation, velocity, angular velocity)
    '''
    return np.array([
        omplState[0][0],
        omplState[0][1],
        omplState[0][2],
        omplState[1].x,
        omplState[1].y,
        omplState[1].z,
        omplState[1].w,
        omplState[2][0],
        omplState[2][1],
        omplState[2][2],
        omplState[3][0],
        omplState[3][1],
        omplState[3][2],
    ])

def copy_state_ompl2pb(pbBodyId, pbClientId, omplState):
    '''Copy the 6DoF state from OMPL into PyBullet state in place
    Args:
        pbBodyId : int
            PyBullet unique object ID of quadrotor body associated with propagator
        pbClientId : int
            ID number of pybullet physics client
        omplState : ob.CompoundState
            ompl's state object for 6DoF with ordering (position, orientation, velocity, angular velocity)

    Refs:
        PyBullet quaternion ordering: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.vy9p26bpc9ft
        OMPL quaternion: https://ompl.kavrakilab.org/classompl_1_1base_1_1SO3StateSpace_1_1StateType.html
    '''
    # define position, orientation, velocity, and angular velocity lists
    p_bu_wu__wu = 3*[None]
    q_bu_wu = 4*[None]
    v_bu_wu__wu = 3*[None]
    w_bu_wu = 3*[None]

    # position
    p_bu_wu__wu[0] = omplState[0][0]
    p_bu_wu__wu[1] = omplState[0][1]
    p_bu_wu__wu[2] = omplState[0][2]

    # orientation
    q_bu_wu[0] = omplState[1].x
    q_bu_wu[1] = omplState[1].y
    q_bu_wu[2] = omplState[1].z
    q_bu_wu[3] = omplState[1].w

    # velocity
    v_bu_wu__wu[0] = omplState[2][0]
    v_bu_wu__wu[1] = omplState[2][1]
    v_bu_wu__wu[2] = omplState[2][2]

    # angular velocity
    w_bu_wu[0] = omplState[3][0]
    w_bu_wu[1] = omplState[3][1]
    w_bu_wu[2] = omplState[3][2]

    # update PyBullet State
    pb.resetBasePositionAndOrientation(
        bodyUniqueId=pbBodyId,
        posObj=p_bu_wu__wu,
        ornObj=q_bu_wu,
        physicsClientId=pbClientId
    )
    pb.resetBaseVelocity(
        objectUniqueId=pbBodyId,
        linearVelocity=v_bu_wu__wu,
        angularVelocity=w_bu_wu,
        physicsClientId=pbClientId
    )

def copy_state_pb2ompl(pbBodyId, pbClientId, omplState):
    '''Copy the 6DoF state from PyBullent into OMPL state in place
    Args:
        pbBodyId : int
            PyBullet unique object ID of quadrotor body associated with propagator
        pbClientId : int
            ID number of pybullet physics client
        omplState : ob.CompoundState
            ompl's state object for 6DoF with ordering (position, orientation, velocity, angular velocity)

    Refs:
        PyBullet quaternion ordering: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.vy9p26bpc9ft
        OMPL quaternion: https://ompl.kavrakilab.org/classompl_1_1base_1_1SO3StateSpace_1_1StateType.html
    '''
    p_bu_wu__wu, q_bu_wu = pb.getBasePositionAndOrientation(
        bodyUniqueId=pbBodyId,
        physicsClientId=pbClientId)
    v_bu_wu__wu, w_bu_wu = pb.getBaseVelocity(
        bodyUniqueId=pbBodyId,
        physicsClientId=pbClientId
    )

    # store state in OMPL CompoundState object in-place
    # position
    omplState[0][0] = p_bu_wu__wu[0]
    omplState[0][1] = p_bu_wu__wu[1]
    omplState[0][2] = p_bu_wu__wu[2]
    # orientation
    omplState[1].x = q_bu_wu[0]
    omplState[1].y = q_bu_wu[1]
    omplState[1].z = q_bu_wu[2]
    omplState[1].w = q_bu_wu[3]
    # velocity
    omplState[2][0] = v_bu_wu__wu[0]
    omplState[2][1] = v_bu_wu__wu[1]
    omplState[2][2] = v_bu_wu__wu[2]
    # angular velocity
    omplState[3][0] = w_bu_wu[0]
    omplState[3][1] = w_bu_wu[1]
    omplState[3][2] = w_bu_wu[2]

_DUMMY_QUADROTORSPACE = QD.QuadrotorStateSpace()

def _pickle_QuadrotorState(state):
    '''pickle QuadrotorState (OMPL compound state) object'''
    px = state[0][0]
    py = state[0][1]
    pz = state[0][2]
    qx = state[1].x
    qy = state[1].y
    qz = state[1].z
    qw = state[1].w
    vx = state[2][0]
    vy = state[2][1]
    vz = state[2][2]
    wx = state[3][0]
    wy = state[3][1]
    wz = state[3][2]
    return _unpickle_QuadrotorState, (px,py,pz,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz)

def _unpickle_QuadrotorState(px,py,pz,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz):
    '''unpickle QuadrotorState (OMPL compound state) object'''
    state = _DUMMY_QUADROTORSPACE.allocState()
    state[0][0] = px
    state[0][1] = py
    state[0][2] = pz
    state[1].x = qx
    state[1].y = qy
    state[1].z = qz
    state[1].w = qw
    state[2][0] = vx
    state[2][1] = vy
    state[2][2] = vz
    state[3][0] = wx
    state[3][1] = wy
    state[3][2] = wz
    return state

def update_pickler_quadrotorstate():
    '''updates pickler to enable pickling and unpickling of ompl objects'''
    copyreg.pickle(_DUMMY_QUADROTORSPACE.allocState().__class__, _pickle_QuadrotorState, _unpickle_QuadrotorState)