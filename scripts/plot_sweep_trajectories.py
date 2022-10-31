"""Plot sweep"""

import os
from functools import partial

import numpy as np

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.simulation.options import SimulationOptions
from farms_core.analysis.plot import plt_farms_style
from farms_core.analysis.metrics import com_positions

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.analysis.plot import plot_trajectories
from farms_amphibious.utils.parse_args import parse_args_sweep


def load_experiment(_sweep_type, exp_data, log, label):
    """Load experiment"""

    # Load
    animat_options_path = os.path.join(log, 'animat_options.yaml')
    animat_options = AmphibiousOptions.load(animat_options_path)
    animat_data = AmphibiousData.from_file(os.path.join(log, 'simulation.hdf5'))
    data_links = animat_data.sensors.links
    timestep = animat_data.timestep
    simulation_options_path = os.path.join(log, 'simulation_options.yaml')
    simulation_options = SimulationOptions.load(simulation_options_path)
    n_iterations = simulation_options.n_iterations
    iteration_0 = 0
    iteration_1 = n_iterations-2  # n_iterations-1

    # Get positions
    sampling = 100  # [Hz]
    positions = com_positions(
        data_links,
        iterations=np.arange(
            start=iteration_0,
            stop=iteration_1,
            step=max(1, round(1/(sampling*timestep))),
            dtype=int,
        ),
    )

    # Data
    if label not in exp_data:
        exp_data[label] = []
    exp_data[label].append({
        'drive': animat_options.control.network.drives[0].initial_value,
        'positions': positions,
    })


def load_data(sweep_type, logs):
    """Load data"""
    exp_data = {}
    for log, label in logs:
        load_experiment(sweep_type, exp_data, log, label)
    return exp_data


def conditional_plot(conditions, function, plot_name, **kwargs):
    """Conditional plot"""
    for condition in conditions:
        function(
            plot_name=plot_name+condition['suffix'],
            condition=condition['condition'],
            **kwargs,
        )


def plot_motion(plots, exp_data):
    """Plot motion"""

    # Conditions
    conditions = [{'suffix': '', 'condition': lambda _: True}]

    # Trajectories
    plot_function = partial(
        conditional_plot,
        conditions=conditions,
        function=plot_trajectories,
        plots=plots,
        exp_data=exp_data,
    )
    plot_function(plot_name='trajectories', legend=True)
    plot_function(plot_name='trajectories_no_legend', legend=False)


def main():
    """Main"""

    # Matplolib options
    plt_farms_style()

    # Clargs
    clargs = parse_args_sweep()

    # Data obtained for plotting
    exp_data = load_data(
        sweep_type=clargs.type,
        logs=zip(clargs.logs, clargs.labels),
    )

    # Plot figure
    plots = {}
    plot_motion(plots=plots, exp_data=exp_data)

    # Save plots
    extension = clargs.extension
    for name, fig in plots.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    profile(main)
