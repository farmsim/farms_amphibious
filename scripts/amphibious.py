#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
from typing import Union

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.simulation.options import Simulator
from farms_mujoco.simulation.simulation import Simulation as MuJoCoSimulation
from farms_sim.utils.parse_args import sim_parse_args
from farms_sim.simulation import (
    setup_from_clargs,
    simulation,
    postprocessing_from_clargs,
)

from farms_amphibious.callbacks import setup_callbacks
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.data.data import (
    AmphibiousData,
    AmphibiousKinematicsData,
    get_amphibious_data,
)
from farms_amphibious.control.network import NetworkODE
from farms_amphibious.control.kinematics import KinematicsController
from farms_amphibious.control.amphibious import (
    AmphibiousController,
    get_amphibious_controller,
)


ENGINE_BULLET = False
try:
    from farms_amphibious.bullet.simulation import (
        AmphibiousPybulletSimulation,
        pybullet_simulation_kwargs,
    )
    ENGINE_BULLET = True
except ImportError as err:
    AmphibiousPybulletSimulation = None
    pybullet_simulation_kwargs = None


def main():
    """Main"""

    # Setup
    pylog.info('Loading options from clargs')
    (
        clargs,
        animat_options,
        sim_options,
        arena_options,
        simulator,
    ) = setup_from_clargs(animat_options_loader=AmphibiousOptions)

    if simulator == Simulator.PYBULLET and not ENGINE_BULLET:
        raise ImportError('Pybullet or farms_bullet not installed')

    # Data
    animat_data: Union[AmphibiousData, AmphibiousKinematicsData] = (
        get_amphibious_data(
            animat_options=animat_options,
            simulation_options=sim_options,
        )
    )

    # Network
    animat_network = NetworkODE(animat_data)

    # Controller
    animat_controller: Union[AmphibiousController, KinematicsController] = (
        get_amphibious_controller(
            animat_data=animat_data,
            animat_network=animat_network,
            animat_options=animat_options,
            sim_options=sim_options,
        )
    )

    # Additional engine-specific options
    options = {}
    if simulator == Simulator.MUJOCO:
        options['callbacks'] = setup_callbacks(animat_options)
    elif simulator == Simulator.PYBULLET:
        options.update(
            pybullet_simulation_kwargs(
                animat_controller=animat_controller,
                animat_options=animat_options,
                sim_options=sim_options,
            )
        )

    # Simulation
    pylog.info('Creating simulation environment')
    sim: Union[MuJoCoSimulation, AmphibiousPybulletSimulation] = simulation(
        animat_data=animat_data,
        animat_options=animat_options,
        animat_controller=animat_controller,
        simulation_options=sim_options,
        arena_options=arena_options,
        simulator=simulator,
        **options,
    )

    # Post-processing
    pylog.info('Running post-processing')
    postprocessing_from_clargs(
        sim=sim,
        clargs=clargs,
        simulator=simulator,
        animat_data_loader=AmphibiousData,
    )


def profile_simulation():
    """Profile simulation"""
    tic = time.time()
    clargs = sim_parse_args()
    profile(function=main, profile_filename=clargs.profile)
    pylog.info('Total simulation time: %s [s]', time.time() - tic)


if __name__ == '__main__':
    profile_simulation()
