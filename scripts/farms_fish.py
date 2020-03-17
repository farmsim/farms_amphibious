#!/usr/bin/env python3
"""Run fish simulation with bullet"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from farms_bullet.experiments.fish.simulation import main as run_sim
from farms_bullet.experiments.fish.simulation import FISH_DIRECTORY
from farms_bullet.animats.amphibious.animat_options import AmphibiousOptions
from farms_bullet.simulations.simulation_options import SimulationOptions


def main():
    """Main"""
    # Animat options
    scale = 1
    animat_options = AmphibiousOptions(
        # collect_gps=True,
        show_hydrodynamics=True,
        scale=scale,
        n_joints_body=20,
        viscous=False,
        resistive=True,
        resistive_coefficients=[
            1e-1*np.array([-1e-4, -5e-1, -3e-1]),
            1e-1*np.array([-1e-6, -1e-6, -1e-6])
        ],
        water_surface=False
    )
    # animat_options.control.drives.forward = 4

    # Simulation options
    simulation_options = SimulationOptions.with_clargs()
    simulation_options.units.meters = 1
    simulation_options.units.seconds = 1e3
    simulation_options.units.kilograms = 1

    fish_name = "crescent_gunnel"
    version_name = "version0"

    # Kinematics
    animat_options.control.kinematics_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        FISH_DIRECTORY,
        fish_name,
        version_name,
        "kinematics",
        "kinematics.csv"
    )
    kinematics = np.loadtxt(animat_options.control.kinematics_file)
    len_kinematics = np.shape(kinematics)[0]
    simulation_options.duration = len_kinematics*1e-2
    pose = kinematics[:, :3]
    # pose *= 1e-3
    # pose *= 1e-3
    # pose[0, :2] *= 1e-3
    # pose[0, 2] *= 1e-3
    position = np.ones(3)
    position[:2] = pose[0, :2]
    orientation = np.zeros(3)
    orientation[2] = pose[0, 2]  #  + np.pi
    velocity = np.zeros(3)
    n_sample = 100
    velocity[:2] = pose[n_sample, :2] - pose[0, :2]
    sampling_timestep = 1e-2
    velocity /= n_sample*sampling_timestep
    kinematics = kinematics[:, 3:]
    kinematics = ((kinematics + np.pi) % (2*np.pi)) - np.pi

    # Walking
    animat_options.spawn.position = position
    animat_options.spawn.orientation = orientation
    animat_options.physics.buoyancy = False
    animat_options.spawn.velocity_lin = velocity
    animat_options.spawn.velocity_ang = [0, 0, 0]
    animat_options.spawn.joints_positions = kinematics[0, :]
    # Swiming
    # animat_options.spawn.position = [-10, 0, 0]
    # animat_options.spawn.orientation = [0, 0, np.pi]

    # Logging
    simulation_options.log_path = "fish_results"

    # Camera options
    simulation_options.video_yaw = 0
    simulation_options.video_pitch = -30
    simulation_options.video_distance = 1
    # simulation_options.video_name = (
    #     "transition_videos/swim2walk_y{}_p{}_d{}".format(
    #         simulation_options.video_yaw,
    #         simulation_options.video_pitch,
    #         simulation_options.video_distance,
    #     )
    # )

    # Run simulation
    sdf_path = os.path.join(
        FISH_DIRECTORY,
        fish_name,
        version_name,
        "sdf",
        "{}.sdf".format(fish_name)
    )
    run_sim(
        sdf_path,
        simulation_options=simulation_options,
        animat_options=animat_options
    )
    # Show results
    plt.show()


def profile():
    """Profile with cProfile"""
    import cProfile
    import pstats
    cProfile.run("main()", "simulation.profile")
    pstat = pstats.Stats("simulation.profile")
    pstat.sort_stats('time').print_stats(30)
    pstat.sort_stats('cumtime').print_stats(30)


def pycall():
    """Profile with pycallgraph"""
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput()):
        main()


if __name__ == '__main__':
    TIC = time.time()
    # main()
    profile()
    # pycall()
    print("Total simulation time: {} [s]".format(time.time() - TIC))
