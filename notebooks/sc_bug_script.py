from pyrmm.setups.quadrotor import QuadrotorPyBulletSetup
quadpb_setup = QuadrotorPyBulletSetup()
cspace = quadpb_setup.space_info.getControlSpace()
c = cspace.allocControl()
c[0] = 0.0
