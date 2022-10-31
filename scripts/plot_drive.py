"""Plot data"""

import os

import numpy as np
import matplotlib.pyplot as plt

from farms_core import pylog
from farms_core.analysis.plot import plt_farms_style, grid
from farms_core.simulation.options import SimulationOptions
from farms_amphibious.data.data import AmphibiousData
# from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.parse_args import parse_args_postprocessing
from farms_amphibious.control.drive import drive_from_config, plot_trajectory


def main():
    """Main"""

    # Style
    plt_farms_style()

    # Clargs
    clargs = parse_args_postprocessing(description='Plot amphibious drive')

    # Load data
    # animat_options = AmphibiousOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AmphibiousData.from_file(clargs.data)
    times = simulation_options.times()
    plots_drive = {}

    # Plot descending drive
    drives = animat_data.network.drives.array
    fig = plt.figure('Drives')
    for drive_i, drive in enumerate(np.array(drives).T):
        plt.plot(times, drive, label=f'drive{drive_i}')
    grid()
    plt.legend()
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
