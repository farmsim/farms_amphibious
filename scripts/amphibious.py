#!/usr/bin/env python3
"""Run salamander simulation with bullet"""

import time
from typing import Union

import farms_pylog as pylog
from farms_data.utils.profile import profile
from farms_data.simulation.options import Simulator
from farms_data.amphibious.data import AmphibiousData
from farms_mujoco.simulation.simulation import Simulation as MuJoCoSimulation
from farms_sim.simulation import (
    setup_from_clargs,
    simulation,
    postprocessing_from_clargs,
)

from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.control.amphibious import AmphibiousController
from farms_amphibious.control.drive import drive_from_config
from farms_amphibious.callbacks import SwimmingCallback
from farms_amphibious.bullet.animat import Amphibious
from farms_amphibious.bullet.simulation import AmphibiousPybulletSimulation
from farms_amphibious.utils.parse_args import sim_parse_args

ENGINE_BULLET = False
try:
    from farms_amphibious.bullet.simulation import AmphibiousPybulletSimulation
    ENGINE_BULLET = True
except ImportError as err:
    pylog.error(err)
    AmphibiousPybulletSimulation = None


def main():
    """Main"""

    # Setup
    pylog.info('Loading options from clargs')
    (
        clargs,
        animat_options,
        sim_options,
        arena_options,
    ) = setup_from_clargs(animat_options_loader=AmphibiousOptions)
    simulator = {
        'MUJOCO': Simulator.MUJOCO,
        'PYBULLET': Simulator.PYBULLET,
    }[clargs.simulator]

    if simulator == Simulator.PYBULLET and not ENGINE_BULLET:
        raise ImportError('Pybullet or farms_bullet not installed')

    # Data
    animat_data = AmphibiousData.from_options(
        animat_options=animat_options,
        simulation_options=sim_options,
    )

    # Controller
    animat_controller = AmphibiousController(
        joints_names=animat_options.control.joints_names(),
        animat_options=animat_options,
        animat_data=animat_data,
        drive=(
            drive_from_config(
                filename=animat_options.control.drive_config,
                animat_data=animat_data,
                simulation_options=sim_options,
            )
            if animat_options.control.drive_config
            else None
        ),
    )

    # Other options
    options = {}

    # Callbacks
    if simulator == Simulator.MUJOCO:
        options['callbacks'] = []
        if animat_options.physics.drag or animat_options.physics.sph:
            options['callbacks'] += [SwimmingCallback(animat_options)]
    elif simulator == Simulator.PYBULLET:
        options['animat'] = Amphibious(
            options=animat_options,
            controller=animat_controller,
            timestep=sim_options.timestep,
            iterations=sim_options.n_iterations,
            units=sim_options.units,
        )
        options['sim_loader'] = AmphibiousPybulletSimulation

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
