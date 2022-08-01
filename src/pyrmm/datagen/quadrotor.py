import hydra
import multiprocess
import time
import torch
import numpy as np
import pybullet as pb
import pybullet_data as pbd

from multiprocess import Manager, Process
from copy import deepcopy
from pathlib import Path
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, instantiate, builds, make_custom_builds_fn
from hydra_zen import ZenField as zf

import os
import sys
from contextlib import contextmanager

import pyrmm.utils.utils as U
import pyrmm.dynamics.quadrotor as QD
from pyrmm.datagen.sampler import sample_risk_metrics
from pyrmm.setups.quadrotor import QuadrotorPyBulletSetup, update_pickler_quadrotorstate


_HASH_LEN = 5
_CONFIG_NAME = "quadrotor_datagen_app"
_SPACE_EXPANSION_FACTOR = 0.15
_SAVE_FNAME = U.format_save_filename(Path(__file__), _HASH_LEN)

##############################################
############# HYDARA-ZEN CONFIGS #############
##############################################

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

_DEFAULT_LIDAR_RANGE = 100.0
_DEFAULT_LIDAR_ANGLES = [
        (np.pi/2, 0),
        (np.pi/2, np.pi/2),
        (np.pi/2, np.pi),
        (np.pi/2, 3*np.pi/2),
        (0,0),
        (np.pi,0)
    ]

QuadrotorPyBulletSetupConfig = pbuilds(QuadrotorPyBulletSetup,
    lidar_range = _DEFAULT_LIDAR_RANGE,
    lidar_angles = _DEFAULT_LIDAR_ANGLES
)

# Default sampler and risk estimator configs
_DEFAULT_N_SAMPLES = 2048
_DEFAULT_DURATION = 2.0
_DEFAULT_N_BRANCHES = 32
_DEFAULT_TREE_DEPTH = 2
_DEFAULT_N_STEPS = 8
_DEFAULT_POLICY = 'uniform_random'

make_config_input = {
    U.SYSTEM_SETUP: QuadrotorPyBulletSetupConfig,
    U.N_SAMPLES: zf(int, _DEFAULT_N_SAMPLES),
    U.DURATION: zf(float, _DEFAULT_DURATION),
    U.N_BRANCHES: zf(int, _DEFAULT_N_BRANCHES),
    U.TREE_DEPTH: zf(int,_DEFAULT_TREE_DEPTH),
    U.N_STEPS: zf(int,_DEFAULT_N_STEPS),
    U.POLICY: zf(str,_DEFAULT_POLICY),
    U.N_CORES: zf(int, multiprocess.cpu_count()),
}
Config = make_config('pcg_rooms_dir', **make_config_input)
ConfigStore.instance().store(_CONFIG_NAME,Config)

##############################################
############### TASK FUNCTIONS ###############
##############################################

@contextmanager
def suppress_stdout():
    '''used to suppress pybullet warnings from spamming stdout'''
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different

def sample_risk_metrics_worker(worker_id, ss_cfg, pcg_room, return_dict, prfx):
    '''functionalized SystemSetup creation and risk metric eval for multiprocessing
    Args:
        worker_id : 
            a dictionary key to uniquely identify worker
        ss_cfg : 
            SystemSetup Config to instantiate SystemSetup
        pcg_room : Path
            path to procedurally generated room environment .world file
        return_dict : multiprocessing.Manager.dict
            dictionary shared between workers to store return values
        prfx : string
            string to add to prefix print statements
    '''

    # instantiate config object to create system setup object
    obj = instantiate(ss_cfg)

    # connect to pb physics client
    pbClientID = pb.connect(pb.DIRECT)

    # load environment
    with suppress_stdout():
        pbObstacleIDs = pb.loadSDF(str(pcg_room))
        pbWallIDs = pb.loadSDF(str(pcg_room.with_suffix('')) + '_walls/model.sdf')

    # extract axes-aligned bounding box from obstacle space
    aabb_min = np.zeros(3)
    aabb_max = np.zeros(3)
    for wall_id in pbWallIDs:
        cur_aabb_min, cur_aabb_max = pb.getAABB(bodyUniqueId=wall_id, physicsClientId=pbClientID)
        aabb_min = np.minimum(aabb_min, cur_aabb_min)
        aabb_max = np.maximum(aabb_max, cur_aabb_max)
    
    # create quadrotor setup state space bounds by expanding obstacle space
    xrange, yrange, zrange = aabb_max - aabb_min
    pxmin = aabb_min[0] - xrange * _SPACE_EXPANSION_FACTOR
    pxmax = aabb_max[0] + xrange * _SPACE_EXPANSION_FACTOR
    pymin = aabb_min[1] - yrange * _SPACE_EXPANSION_FACTOR
    pymax = aabb_max[1] + yrange * _SPACE_EXPANSION_FACTOR
    pzmin = aabb_min[2] - zrange * _SPACE_EXPANSION_FACTOR
    pzmax = aabb_max[2] + zrange * _SPACE_EXPANSION_FACTOR

    # finish instantiating quadrotor pybullet setup object with custom state bounds
    quadpb_setup = getattr(obj, U.SYSTEM_SETUP)(
        pb_client_id = pbClientID,
        pxmin = pxmin,
        pxmax = pxmax,
        pymin = pymin,
        pymax = pymax,
        pzmin = pzmin,
        pzmax = pzmax,
    )

    # sample states in environment and compute risk metrics
    # Note: don't run sample_risk_metric in multiprocess mode
    # since multi-processing is brokered in outer loop when using pybullet
    risk_data = sample_risk_metrics(sysset=quadpb_setup, cfg_obj=obj, multiproc=False, prfx=prfx)

    # store result, check for collision in key
    assert worker_id not in return_dict
    return_dict[worker_id] = risk_data

    # disconnect from physics client
    pb.disconnect(pbClientID)



@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config): 
    ''' Instantiate QuadrotorPyBullet setup and sample risk metric data'''

    # update pickler to enable parallelization of OMPL objects
    update_pickler_quadrotorstate()

    t_start = time.time()

    # get the proc gen room environments for state sampling
    pcg_rooms = list(Path(U.get_abs_path_str(cfg.pcg_rooms_dir)).glob('*.world'))
    n_pcg_rooms = len(pcg_rooms)

    # split processing into fixed number of parallel processes
    n_jobs = getattr(cfg, U.N_CORES)
    n_total_samples = getattr(cfg, U.N_SAMPLES)
    n_samples_per_job = U.min_linfinity_int_vector(n_jobs, n_total_samples)

    # iterate through all room environment sequentially, splitting the
    # n_samples per environment into parallel tasks
    for pcgr_num, pcgr in enumerate(pcg_rooms):

        # get save filename 
        save_name = _SAVE_FNAME + '_' + pcgr.stem

        print('\nStarting Room {} ({}/{}) DataGen\n'.format(pcgr.stem, pcgr_num+1, n_pcg_rooms))

        # split n_samples into parallel task
        manager = Manager()
        return_dict = manager.dict()
        jobs = n_jobs*[None]
        for wrkid, n_cur_samples in enumerate(n_samples_per_job):

            # create a config with subset of samples
            cur_cfg = deepcopy(cfg)
            setattr(cur_cfg, U.N_SAMPLES, n_cur_samples)

            # call helper function to instantiate quad system setup and get risk data
            prfx = 'Env {}/{} | Job {}/{}: '.format(pcgr_num+1, n_pcg_rooms, wrkid+1,n_jobs)
            p = Process(target=sample_risk_metrics_worker, args=(wrkid, cur_cfg, pcgr, return_dict, prfx))
            jobs[wrkid] = p
            p.start()

        # join processes once they complete
        for proc in jobs:
            proc.join()

        # compile data and save
        print('\nRoom {} ({}/{}) DataGen Complete\n'.format(pcgr.stem, pcgr_num+1, n_pcg_rooms))
        comp_risk_data = sum([list(dat) for dat in return_dict.values()], [])
        torch.save(comp_risk_data, open(save_name+".pt", "wb"))
    
    print("\n\n~~~COMPLETE~~~\n\nTotal elapsed time: {:.2f}".format(time.time()-t_start))


if __name__ == "__main__":
    task_function()




