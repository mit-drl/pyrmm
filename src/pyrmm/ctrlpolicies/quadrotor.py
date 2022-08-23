'''Control Policies for Quadrotor System with Thrust-Torque Control Space

Refs: 
    + http://docs.px4.io/main/en/flight_stack/controller_diagrams.html
    + https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf
    + http://docs.px4.io/main/en/config_mc/pid_tuning_guide_multicopter.html#rate-controller
    + https://github.com/PX4/PX4-Autopilot/blob/main/src/modules/mc_pos_control/PositionControl/PositionControl.cpp
    + https://github.com/PX4/PX4-Autopilot/blob/main/src/modules/mc_pos_control/PositionControl/ControlMath.cpp
    + https://github.com/PX4/PX4-Autopilot/blob/main/src/modules/mc_att_control/mc_att_control_main.cpp
    + https://github.com/PX4/PX4-Autopilot/blob/main/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp
'''

class AttitudeController():
    '''Computes angular rate setpoints based on attitude quaternion'''
    def __init__(self, ) -> None:
        pass

