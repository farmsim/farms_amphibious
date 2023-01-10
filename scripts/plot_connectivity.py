"""Plot connectivity"""

import os
import argparse

from farms_core import pylog
from farms_core.analysis.plot import plt_farms_style

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.network import plot_networks_maps


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

    # Style
    plt_farms_style()

    # Clargs
    clargs = parse_args()

    # Load data
    animat_options = AmphibiousOptions.load(clargs.animat)
    animat_data = AmphibiousData.from_file(clargs.data)

    # Plot connectivity
    plots_network = (
        plot_networks_maps(
            data=animat_data,
            animat_options=animat_options,
            show_all=False,
        )[1]
        if animat_options.morphology.n_dof_legs <= 4
        else {}
    )

    # Save plots
    extension = 'pdf'
    for name, fig in plots_network.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
