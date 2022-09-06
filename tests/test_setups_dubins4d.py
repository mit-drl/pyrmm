from pyrmm.setups.dubins4d import Dubins4DReachAvoidSetup
from pyrmm.environments.dubins4d_reachavoid import Dubins4dReachAvoidEnv

def test_Dubins4DReachAvoid_init_0():
    
    # ~~~ ARRANGE ~~~
    # create default environment
    env = Dubins4dReachAvoidEnv()

    # ~~~ ACT ~~~
    # create dubins4d reach avoid setup
    d4d_setup = Dubins4DReachAvoidSetup(env = env)

    # ~~~ ASSERT ~~~
    pass