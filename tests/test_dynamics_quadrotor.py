import pytest
import pathlib
import numpy as np
import pybullet as pb

import pyrmm.dynamics.quadrotor as QD

QUAD_URDF = str(pathlib.Path(__file__).parent.absolute().joinpath("quadrotor.urdf"))

def test_QuadrotorStateSpace_bounds_0():
    '''check that state space bounds are set as intended'''
    # ~~ ARRANGE ~~
    # random yet fixed position, velocity, and ang vel bounds
    bounds = dict()
    bounds['pos_low']= [-2.55944188, -3.08836728, -2.58956761]
    bounds['pos_high'] = [13.58335454, 11.78403468,  5.68640255]
    bounds['vel_low']= [-4.34878973,  0.99390659,  1.31070602]
    bounds['vel_high'] = [6.0623723 , 8.94204147, 9.349446  ]
    bounds['omg_low']= [-0.85770383,  4.52460971, -2.74973743]
    bounds['omg_high'] = [11.15227062,  9.96662472, 10.04022507]

    # ~~ ACT ~~
    # create state space with bounds
    quadspace = QD.QuadrotorStateSpace(bounds=bounds)

    # ~~ ASSERT ~~~
    pos_bounds = quadspace.getSubspace(0).getBounds()
    assert np.isclose(pos_bounds.low[0], bounds['pos_low'][0])
    assert np.isclose(pos_bounds.low[1], bounds['pos_low'][1])
    assert np.isclose(pos_bounds.low[2], bounds['pos_low'][2])
    assert np.isclose(pos_bounds.high[0], bounds['pos_high'][0])
    assert np.isclose(pos_bounds.high[1], bounds['pos_high'][1])
    assert np.isclose(pos_bounds.high[2], bounds['pos_high'][2])

    vel_bounds = quadspace.getSubspace(2).getBounds()
    assert np.isclose(vel_bounds.low[0], bounds['vel_low'][0])
    assert np.isclose(vel_bounds.low[1], bounds['vel_low'][1])
    assert np.isclose(vel_bounds.low[2], bounds['vel_low'][2])
    assert np.isclose(vel_bounds.high[0], bounds['vel_high'][0])
    assert np.isclose(vel_bounds.high[1], bounds['vel_high'][1])
    assert np.isclose(vel_bounds.high[2], bounds['vel_high'][2])

    omg_bounds = quadspace.getSubspace(3).getBounds()
    assert np.isclose(omg_bounds.low[0], bounds['omg_low'][0])
    assert np.isclose(omg_bounds.low[1], bounds['omg_low'][1])
    assert np.isclose(omg_bounds.low[2], bounds['omg_low'][2])
    assert np.isclose(omg_bounds.high[0], bounds['omg_high'][0])
    assert np.isclose(omg_bounds.high[1], bounds['omg_high'][1])
    assert np.isclose(omg_bounds.high[2], bounds['omg_high'][2])
    
def test_QuadrotorStateSpace_bounds_sample_0():
    '''check that state space with zero-volume bounds sample a deterministic number'''
    # ~~ ARRANGE ~~
    # random yet fixed zero-volume bounds
    bounds = dict()
    bounds['pos_low']= [-99.68020119,  53.3656405 ,  77.48463094]
    bounds['pos_high'] = [-99.68020119,  53.3656405 ,  77.48463094]
    bounds['vel_low']= [-5.74721878,  4.86715528, -5.42107527]
    bounds['vel_high'] = [-5.74721878,  4.86715528, -5.42107527]
    bounds['omg_low']= [ 6.75674113,  7.69125197, -4.14222264]
    bounds['omg_high'] = [ 6.75674113,  7.69125197, -4.14222264]

    # ~~ ACT ~~
    # create state space with bounds
    quadspace = QD.QuadrotorStateSpace(bounds=bounds)
    ssampler = quadspace.allocStateSampler()
    sstate = quadspace.allocState()
    ssampler.sampleUniform(sstate)

    # ~~ ASSERT ~~~
    assert np.isclose(sstate[0][0], bounds['pos_low'][0])
    assert np.isclose(sstate[0][1], bounds['pos_low'][1])
    assert np.isclose(sstate[0][2], bounds['pos_low'][2])
    assert np.isclose(sstate[2][0], bounds['vel_low'][0])
    assert np.isclose(sstate[2][1], bounds['vel_low'][1])
    assert np.isclose(sstate[2][2], bounds['vel_low'][2])
    assert np.isclose(sstate[3][0], bounds['omg_low'][0])
    assert np.isclose(sstate[3][1], bounds['omg_low'][1])
    assert np.isclose(sstate[3][2], bounds['omg_low'][2])
    q = np.array([sstate[1].x, sstate[1].y, sstate[1].z, sstate[1].w])
    assert np.isclose(np.dot(q,q), 1.0)

if __name__ == "__main__":
    # test_copy_state_2(None)
    # test_update_pickler_quadrotorstate_0(None)
    pass