"""Network"""

import os

import numpy as np
import matplotlib.pyplot as plt

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.simulation.options import SimulationOptions
from farms_sim.utils.parse_args import sim_argument_parser
from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.control.network import NetworkODE
from farms_amphibious.control.amphibious import get_amphibious_controller
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.network import plot_networks_maps


def network_parse_args():
    """Parse arguments"""
    parser = sim_argument_parser()
    parser.add_argument(
        '--show_plots',
        action='store_true',
        help='Show plots',
    )
    parser.add_argument(
        '--show_connectivity_maps',
        action='store_true',
        help='Show connectivity maps',
    )
    args, _ = parser.parse_known_args()
    return args


def run_simulation(controller, n_iterations, timestep):
    """Run simulation"""
    for iteration in range(n_iterations-1):
        controller.step(iteration, iteration*timestep, timestep)


def analysis(data, times, morphology, **kwargs):
    """Analysis"""
    # Plot data
    if kwargs.pop('show_plots', False):
        # data.plot(times)
        data.state.plot_phases(times)
        data.state.plot_amplitudes(times)

    # Network
    if kwargs.pop('show_connectivity_maps', False):
        sep = '\n  - '
        pylog.info(
            'Oscillator connectivity information\n%s',
            sep.join([
                f'O_{connection[0]} <- O_{connection[1]}'
                f' (w={weight}, theta={phase})'
                for connection, weight, phase in zip(
                    data.network.osc_connectivity.connections.array,
                    data.network.osc_connectivity.weights.array,
                    data.network.osc_connectivity.desired_phases.array,
                )
            ])
        )
        pylog.info(
            'Contacts connectivity information\n%s',
            sep.join([
                f'O_{connection[0]} <- contact_{connection[1]}'
                f' (frequency_gain={weight})'
                for connection, weight in zip(
                    data.network.contacts_connectivity.connections.array,
                    data.network.contacts_connectivity.weights.array,
                )
            ])
        )
        pylog.info(
            'Xfrc connectivity information\n%s',
            sep.join([
                f'O_{connection[0]} <- link_{connection[1]}'
                f' (type={connection[2]}, weight={weight})'
                for connection, weight in zip(
                    data.network.xfrc_connectivity.connections.array,
                    data.network.xfrc_connectivity.weights.array,
                )
            ])
        )
        sep = '\n'
        pylog.info(
            sep.join([
                'Network infromation:',
                '  - Oscillators:',
                '     - Intrinsic frequencies: {}',
                '     - Nominal amplitudes: {}',
                '     - Rates: {}',
                '  - Connectivity shape: {}',
                '  - Contacts connectivity shape: {}',
                '  - Xfrc connectivity shape: {}',
            ]),
            np.shape(data.network.oscillators.intrinsic_frequencies.array),
            np.shape(data.network.oscillators.nominal_amplitudes.array),
            np.shape(data.network.oscillators.rates.array),
            np.shape(data.network.osc_connectivity.connections.array),
            np.shape(data.network.contacts_connectivity.connections.array),
            np.shape(data.network.xfrc_connectivity.connections.array),
        )

        plot_networks_maps(morphology, data)


def main(clargs=None, filename='data.hdf5'):
    """Main"""

    # Setup
    if clargs is None:
        clargs = network_parse_args()
    animat_options = AmphibiousOptions.load(filename=clargs.animat_config)
    sim_options = SimulationOptions.load(filename=clargs.simulation_config)

    # Animat data
    animat_data = AmphibiousData.from_options(
        animat_options=animat_options,
        simulation_options=sim_options,
    )

    # Animat network
    animat_network = NetworkODE(data=animat_data)

    # Animat controller
    animat_controller = get_amphibious_controller(
        animat_data=animat_data,
        animat_network=animat_network,
        animat_options=animat_options,
        sim_options=sim_options,
    )

    # Run simulation
    profile(
        run_simulation,
        controller=animat_controller,
        n_iterations=sim_options.n_iterations,
        timestep=sim_options.timestep,
        profile_filename=clargs.profile,
    )

    # Save data
    output_path = os.path.join(clargs.log_path, filename)
    pylog.debug('Saving data to %s', output_path)
    animat_data.to_file(filename=output_path)
    pylog.debug('Save complete')

    # Post-processing
    analysis(
        data=animat_data,
        times=np.arange(
            start=0,
            stop=sim_options.n_iterations*sim_options.timestep,
            step=sim_options.timestep,
        ),
        morphology=animat_options.morphology,
        show_plots=clargs.show_plots,
        show_connectivity_maps=clargs.show_connectivity_maps,
    )

    # Show
    plt.show()


if __name__ == '__main__':
    main()
