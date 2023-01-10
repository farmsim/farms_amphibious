"""Plot connectivity matrix"""

import os

from farms_core import pylog
from farms_core.analysis.plot import plt_farms_style

# from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.network import plot_connectivity_matrix
from farms_amphibious.utils.parse_args import parse_args_postprocessing


def main():
    """Main"""

    # Style
    plt_farms_style()

    # Clargs
    clargs = parse_args_postprocessing()

    # Load data
    animat_options = AmphibiousOptions.load(clargs.animat)
    # animat_data = AmphibiousData.from_file(clargs.data)

    # Plot connectivity
    plots_network = plot_connectivity_matrix(animat_options=animat_options)

    # Save plots
    extension = 'pdf'
    for name, fig in plots_network.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
