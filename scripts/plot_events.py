"""Extract steps timings"""

import numpy as np
import matplotlib.pyplot as plt

from farms_core.io.yaml import yaml2pyobject
from farms_core.simulation.options import SimulationOptions
from farms_core.analysis.plot import plt_farms_style, save_plots

from farms_amphibious.data.data import AmphibiousData
# from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.parse_args import parser_postprocessing


def main():
    """Main"""

    # Style
    plt_farms_style()

    # Clargs
    parser = parser_postprocessing(description='Plot amphibious events')
    parser.add_argument(
        '--analysis',
        type=str,
        help='Analysis results',
    )
    clargs = parser.parse_args()

    # Load data
    # animat_options = AmphibiousOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AmphibiousData.from_file(clargs.data)
    n_iterations = simulation_options.n_iterations

    # Plot simulation data
    times = simulation_options.times()
    assert len(times) == n_iterations, f'{len(times)=} != {n_iterations=}'
    times = times[:animat_data.sensors.links.array.shape[0]]

    # Get data
    plots_sim = {}
    analysis = yaml2pyobject(filename=clargs.analysis)
    joints_pos = np.array(animat_data.sensors.joints.positions_all())

    # Plot
    suffixes = ['body', 'limb']
    for i, (joint_index, peaks) in enumerate(analysis['joint_max_pos'].items()):
        for data, ylabel, suffix in [
                [joints_pos, 'Position [rad]', '_pos_max'],
                # [joints_vel, 'Velocitiy [rad/s]', '_velocity'],
        ]:
            name = f'events_joint{suffix}_{suffixes[i]}'
            fig = plt.figure(name)
            plots_sim[name] = fig
            plt.plot(times, data[:, joint_index])
            plt.plot(times[peaks], data[peaks, joint_index], "x")
            # plt.plot(np.zeros_like(data), "--", color="gray")
            plt.xlabel('Time [s]')
            plt.ylabel(ylabel)

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
