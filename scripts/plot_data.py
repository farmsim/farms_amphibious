"""Plot data"""

import numpy as np

from farms_core.simulation.options import SimulationOptions
from farms_core.analysis.plot import plt_farms_style, save_plots
from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.utils.parse_args import parse_args_postprocessing


def main():
    """Main"""

    # Style
    plt_farms_style()

    # Clargs
    clargs = parse_args_postprocessing(description='Plot amphibious data')

    # Load data
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AmphibiousData.from_file(clargs.data)
    n_iterations = simulation_options.n_iterations
    timestep = animat_data.timestep

    # Plot simulation data
    times = np.arange(start=0, stop=timestep*(n_iterations-0.5), step=timestep)
    assert len(times) == n_iterations, f'{len(times)=} != {n_iterations=}'
    times = times[:animat_data.sensors.links.array.shape[0]]

    # Plot sensors
    plots_sim = animat_data.plot(times)

    # Save plots
    save_plots(
        plots=plots_sim,
        path=clargs.output,
        extension='pdf',
        bbox_inches='tight',
        dpi=300,
    )


if __name__ == '__main__':
    main()
