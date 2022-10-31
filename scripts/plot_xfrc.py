"""Plot xfrc"""

import numpy as np
import matplotlib.pyplot as plt

from farms_core.simulation.options import SimulationOptions
from farms_core.analysis.plot import plt_farms_style, save_plots, colorgraph

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.model.convention import AmphibiousConvention
from farms_amphibious.utils.parse_args import parse_args_postprocessing


def main():
    """Main"""

    # Style
    plt_farms_style()

    # Clargs
    clargs = parse_args_postprocessing(description='Plot amphibious xfrc')

    # Load data
    animat_options = AmphibiousOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AmphibiousData.from_file(clargs.data)
    n_iterations = simulation_options.n_iterations

    # Plot simulation data
    times = simulation_options.times()
    assert len(times) == n_iterations, f'{len(times)=} != {n_iterations=}'
    times = times[:animat_data.sensors.links.array.shape[0]]

    # Plot xfrc positions
    xfrc = np.array(animat_data.sensors.xfrc.forces())
    xfrc_x = xfrc[:, :, 0]
    xfrc_y = xfrc[:, :, 1]
    xfrc_z = xfrc[:, :, 2]
    labels = animat_data.sensors.xfrc.names

    # Convention
    convention = AmphibiousConvention.from_amphibious_options(animat_options)
    n_xfrc = len(labels)
    all_indices = list(range(n_xfrc))
    body_indices = list(range(convention.n_links_body()))
    legs_indices = list(range(
        convention.n_links_body(),
        convention.n_links_body()+convention.n_links_legs(),
    ))

    # Plot
    plots_sim = {}
    for data, clabel, suffix in [
            [xfrc_x, 'Force [N]', '_x'],
            [xfrc_y, 'Force [N]', '_y'],
            [xfrc_z, 'Force [N]', '_z'],
    ]:
        for cmap, suffix2 in [['cividis', '']]:
            for suffix3, indices, aspect in [
                    ['_all', all_indices, 1.0],
                    ['_body', body_indices, 1.0],
                    ['_legs', legs_indices, 1.0],
            ]:
                fig = plt.figure(f'colorgraph_xfrc{suffix}{suffix2}{suffix3}')
                plots_sim[f'colorgraph_xfrc{suffix}{suffix2}{suffix3}'] = fig
                if not indices:
                    continue
                max_val = 1.1*np.percentile(np.abs(data.flatten()), 90)
                colorgraph(
                    data=data.T[indices, :],
                    labels=np.array(labels)[indices],
                    n_pixel_y=4,
                    gap=1,
                    vmin=-max_val,
                    vmax=+max_val,
                    x_extent=[0, times[-1]],
                    cmap=cmap,
                    xlabel='Time [s]',
                    ylabel='Sensors',
                    clabel=clabel,
                    aspect=aspect,
                )

    # Save plots
    save_plots(
        plots=plots_sim,
        path=clargs.output,
        extension='pdf',
        bbox_inches='tight',
        dpi=600,
    )


if __name__ == '__main__':
    main()
