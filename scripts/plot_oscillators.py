"""Plot oscillators"""

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
    clargs = parse_args_postprocessing(description='Plot amphibious oscillators')

    # Load data
    animat_options = AmphibiousOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AmphibiousData.from_file(clargs.data)
    n_iterations = simulation_options.n_iterations
    timestep = animat_data.timestep

    # Plot simulation data
    times = simulation_options.times()
    assert len(times) == n_iterations, f'{len(times)=} != {n_iterations=}'
    times = times[:animat_data.sensors.links.array.shape[0]]

    # Oscillators data
    phases = np.array(animat_data.state.phases_all())
    oscillators_phs = phases % (2*np.pi)
    oscillators_amp = np.array(animat_data.state.amplitudes_all())
    oscillators_out = np.array(animat_data.state.outputs_all())
    labels = animat_data.network.oscillators.names
    oscillators_frq = np.diff(phases, axis=0)/(2*np.pi*timestep)

    # Convention
    convention = AmphibiousConvention.from_amphibious_options(animat_options)
    n_oscillators = len(labels)
    all_indices = list(range(n_oscillators))
    n_osc_body = convention.n_osc_body()
    n_osc_legs = convention.n_osc_legs()
    body_indices = list(range(n_osc_body))
    body_left_indices =  [i for i in body_indices if not i%2]
    body_right_indices =  [i for i in body_indices if i%2]
    legs_indices = list(range(n_osc_body, n_osc_body+n_osc_legs))

    # Plot
    plots_sim = {}
    for data, clabel, suffix in [
            [oscillators_frq, 'Frequencies [Hz]', '_frequencies'],
            [oscillators_phs, 'Phase [rad]', '_phases'],
            [oscillators_amp, 'Amplitude', '_amplitudes'],
            [oscillators_out, 'Output', '_outputs'],
    ]:
        for cmap, suffix2 in [
                ['cividis', ''],
                # ['viridis', '_viridis'],
                # ['BrBG', '_brbg'],
                # ['turbo', '_turbo'],
                # ['GnBu', '_gnbu'],
                # ['hot_r', '_hotr'],
                # ['gist_heat_r', '_gistheatr'],
        ]:
            for suffix3, indices, aspect in [
                    ['_all', all_indices, 1.0],
                    ['_body', body_indices, 1.0],
                    ['_bodyl', body_left_indices, 1.0],
                    ['_bodyr', body_right_indices, 1.0],
                    ['_legs', legs_indices, 1.0],
            ]:
                fig = plt.figure(f'colorgraph_oscillators{suffix}{suffix2}{suffix3}')
                plots_sim[f'colorgraph_oscillators{suffix}{suffix2}{suffix3}'] = fig
                if not indices:
                    continue
                colorgraph(
                    data=data.T[indices, :],
                    labels=np.array(labels)[indices],
                    n_pixel_y=4,
                    gap=1,
                    vmin=np.percentile(data[:, indices].flatten(), 1),
                    vmax=np.percentile(data[:, indices].flatten(), 99),
                    x_extent=[0, times[-1]],
                    cmap=cmap,
                    xlabel='Time [s]',
                    ylabel='Oscillators',
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
