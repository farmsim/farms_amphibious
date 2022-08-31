"""Plot data"""

import os
import argparse

import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt

from farms_core import pylog
from farms_core.simulation.options import SimulationOptions
from farms_amphibious.data.data import AmphibiousData
# from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.control.drive import drive_from_config, plot_trajectory

plt.rc('axes', prop_cycle=(
    cycler(linestyle=['-', '--', '-.', ':'])
    * cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])
))
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Palatino'],
})


def parse_args():
    """Parse args"""
    parser = argparse.ArgumentParser(
        description='Plot amphibious simulation data',
        formatter_class=(
            lambda prog:
            argparse.HelpFormatter(prog, max_help_position=50)
        ),
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Data',
    )
    parser.add_argument(
        '--animat',
        type=str,
        help='Animat options',
    )
    parser.add_argument(
        '--simulation',
        type=str,
        help='Simulation options',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path',
    )
    parser.add_argument(
        '--drive_config',
        type=str,
        default='',
        help='Descending drive method',
    )
    return parser.parse_args()


def main():
    """Main"""

    # Clargs
    clargs = parse_args()

    # Load data
    # animat_options = AmphibiousOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AmphibiousData.from_file(clargs.data)
    n_iterations = simulation_options.n_iterations
    timestep = animat_data.timestep
    times = np.arange(start=0, stop=timestep*n_iterations, step=timestep)
    plots_drive = {}

    # Plot descending drive
    drives = animat_data.network.drives.array
    fig = plt.figure('Drives')
    for drive_i, drive in enumerate(np.array(drives).T):
        plt.plot(times, drive, label=f'drive{drive_i}')
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Drive value')
    plots_drive['drives'] = fig

    # Plot trajectory
    if clargs.drive_config:
        pos = np.array(animat_data.sensors.links.urdf_positions()[:, 0])
        drive = drive_from_config(
            filename=clargs.drive_config,
            animat_data=animat_data,
            simulation_options=simulation_options,
        )
        fig3 = plot_trajectory(drive.strategy, pos)
        plots_drive['trajectory'] = fig3

    # Save plots
    extension = 'pdf'
    for name, fig in plots_drive.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
