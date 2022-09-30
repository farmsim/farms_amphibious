"""Plot sweep"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

from farms_core import pylog
from farms_core.analysis.plot import grid, plt_latex_options
from farms_core.utils.profile import profile
from farms_core.simulation.options import SimulationOptions
from farms_core.analysis.metrics import (
    compute_torque_integral,
    average_2d_velocity,
)

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.analysis.metrics import analyse_gait_amphibious


plt_latex_options()


def parse_args():
    """Parse args"""
    parser = argparse.ArgumentParser(
        description='Plot amphibious sweep',
        formatter_class=(
            lambda prog:
            argparse.HelpFormatter(prog, max_help_position=50)
        ),
    )
    parser.add_argument(
        '--type',
        type=str,
        help='Sweep type',
    )
    parser.add_argument(
        '--extension',
        type=str,
        help='Output extension',
    )
    parser.add_argument(
        '-l', '--logs',
        metavar='log1 log2 ...',
        type=str,
        nargs='+',
        default=[],
        help='Sweep logs folders',
    )
    parser.add_argument(
        '--labels',
        metavar='label1 label2 ...',
        type=str,
        nargs='+',
        default=[],
        help='Sweep experiments labels',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path',
    )
    return parser.parse_args()


def load_experiment(sweep_type, exp_data, log, label):
    """Load experiment"""

    # Load
    animat_options_path = os.path.join(log, 'animat_options.yaml')
    animat_options = AmphibiousOptions.load(animat_options_path)
    data_path = os.path.join(log, 'simulation.hdf5')
    animat_data = AmphibiousData.from_file(data_path)
    data_links = animat_data.sensors.links
    timestep = animat_data.timestep
    simulation_options_path = os.path.join(log, 'simulation_options.yaml')
    simulation_options = SimulationOptions.load(simulation_options_path)
    n_iterations = simulation_options.n_iterations
    times = simulation_options.times()
    iteration_0 = n_iterations//4
    iteration_1 = n_iterations-2  # n_iterations-1
    assert len(times) == n_iterations, f'{len(times)=} != {n_iterations=}'
    times = times[:animat_data.sensors.links.array.shape[0]]

    if sweep_type == 'drive':

        # Get drive
        drive = animat_options.control.network.drives[0].initial_value

        # Get velocity
        link_id = 0
        positions_all = np.array(data_links.urdf_positions()[:, link_id, :])

        # Torques integral
        data_joints = animat_data.sensors.joints
        torque_integrals = [
            compute_torque_integral(
                data_joints=data_joints,
                iteration0=iteration_0,
                iteration1=iteration_1,
                exponent=exponent,
                times=times,
                timestep=timestep
            )
            for exponent in range(1, 5)
        ]

        # Average velocity
        average_velocity = average_2d_velocity(
            positions=positions_all,
            iterations=[iteration_0, iteration_1],
            timestep=timestep,
        )

        # Data
        if label not in exp_data:
            exp_data[label] = []
        metrics = {
            'drive': drive,
            'average_velocity': average_velocity,
            'positions': positions_all[:, :2],
            'n_legs': animat_options.morphology.n_legs,
        }
        metrics.update({
            f'torque_integral{1+i}': torque_integrals[i]
            for i in range(4)
        })
        metrics.update({
            f'cot{i+1}': torque_integrals[i]/average_velocity
            for i in range(4)
        })
        exp_data[label].append(metrics)

        # Gaits
        gait = analyse_gait_amphibious(animat_data, animat_options)
        for key, value in gait.items():
            exp_data[label][-1][f'gait_{key}'] = value


def load_data(sweep_type, logs):
    """Load data"""
    exp_data = {}
    for log, label in logs:
        load_experiment(sweep_type, exp_data, log, label)
    return exp_data


def plot_trajectories(plots, exp_data, **kwargs):
    """Plot trajectories"""
    plot_name = kwargs.pop('plot_name')
    condition = kwargs.pop('condition', lambda _: True)
    plots[plot_name] = plt.figure(plot_name)
    for label_i, (label, data) in enumerate(exp_data.items()):
        prefix = f'{label} - ' if len(exp_data) > 1 else ''
        for exp in data:
            if condition(exp['drive']):
                plt.plot(
                    exp['positions'][:, 0],
                    exp['positions'][:, 1],
                    label=f'{prefix}Drive={exp["drive"]}',
                    zorder=10-label_i,
                )
    plt.legend()
    plt.xlabel('Position X [m]')
    plt.ylabel('Position Y [m]')
    grid()
    plt.gca().set_aspect('equal')


def plot_element(plots, exp_data, **kwargs):
    """Plot element"""
    plot_name = kwargs.pop('plot_name')
    xdata = kwargs.pop('xdata')
    ydata = kwargs.pop('ydata')
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    ylim = kwargs.pop('ylim', None)
    zorder_i = kwargs.pop('zorder_i', 0)
    label_template = kwargs.pop('label_template', '{label}')
    show_legend = kwargs.pop('show_legend', False)
    condition = kwargs.pop('condition', lambda _: True)
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = '-'
    if 'marker' not in kwargs:
        kwargs['marker'] = '.'
    plots[plot_name] = plt.figure(plot_name)
    for label_i, (label, data) in enumerate(exp_data.items()):
        plt.plot(
            [exp[xdata] for exp in data if condition(exp['drive'])],
            [exp[ydata] for exp in data if condition(exp['drive'])],
            label=label_template.format(label=label),
            zorder=10*(10-label_i)-zorder_i,
            **kwargs,
        )
    if show_legend or len(exp_data) > 1:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    grid()


def plot_multi_exponent(plots, exp_data, **kwargs):
    """Plot multi-exponent"""
    plot_name_template = kwargs.pop('plot_name')
    xdata = kwargs.pop('xdata')
    ydata = kwargs.pop('ydata')
    xlabel = kwargs.pop('xlabel')
    ylabel = kwargs.pop('ylabel')
    equation = kwargs.pop('equation')
    show_legend = kwargs.pop('show_legend', False)
    condition = kwargs.pop('condition', lambda _: True)
    if 'linestyle' not in kwargs:
        kwargs['linestyle'] = '-'
    if 'marker' not in kwargs:
        kwargs['marker'] = '.'
    for exponent in range(1, 5):
        plot_name = plot_name_template.format(exponent)
        plots[plot_name] = plt.figure(plot_name)
        for label_i, (label, data) in enumerate(exp_data.items()):
            plt.plot(
                [
                    exp[xdata]
                    for exp in data
                    if condition(exp['drive'])
                ],
                [
                    exp[ydata.format(exponent)]
                    for exp in data
                    if condition(exp['drive'])
                ],
                label=label,
                zorder=10-label_i,
                **kwargs,
            )
        if show_legend or len(exp_data) > 1:
            plt.legend()
        plt.xlabel(xlabel)
        exp = f'^{exponent}' if exponent > 1 else ''
        plt.ylabel(ylabel.format(equation=equation.format(exp=exp)))
        grid()


def conditional_plot(conditions, function, plot_name, **kwargs):
    """Conditional plot"""
    for condition in conditions:
        function(
            plot_name=plot_name+condition['suffix'],
            condition=condition['condition'],
            **kwargs,
        )


def plot_drive(plots, exp_data):
    """Plot drive"""

    # Style
    plt.style.use('tableau-colorblind10')

    # Conditions
    conditions = [
        {
            'suffix': '',
            'condition': lambda _: True,
        },
        {
            'suffix': '_wlk',
            'condition': lambda drive: drive <= 3,
        },
        {
            'suffix': '_swm',
            'condition': lambda drive: drive > 3,
        }
    ]

    # Velocities
    conditional_plot(
        conditions=conditions,
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='velocities',
        xdata='drive',
        ydata='average_velocity',
        xlabel='Drive',
        ylabel='Velocity [m/s]',
    )

    # Equations
    equation_torque_integral = (
        r'$\displaystyle'
        r' \frac{{1}}{{T_1 - T_0}}'
        r' \int_{{T_0}}^{{T_1}}'
        r' \sum_{{j=1}}^N |\tau_{{t,j}}|{exp}'
        r' dt$'
    )
    equation_cot = (
        r'$\displaystyle'
        r' \frac{{1}}{{\bar{{v}}}}'
        r' \frac{{1}}{{T_1 - T_0}}'
        r' \int_{{T_0}}^{{T_1}}'
        r' \sum_{{j=1}}^N |\tau_{{t,j}}|{exp}'
        r' dt$'
    )

    # Torque integral
    conditional_plot(
        conditions=conditions,
        function=plot_multi_exponent,
        plots=plots,
        exp_data=exp_data,
        plot_name='torque_integral{}',
        xdata='drive',
        ydata='torque_integral{}',
        xlabel='Drive',
        equation=equation_torque_integral,
        ylabel='{equation}',
    )

    # Cost of transport
    conditional_plot(
        conditions=conditions,
        function=plot_multi_exponent,
        plots=plots,
        exp_data=exp_data,
        plot_name='cot{}',
        xdata='drive',
        ydata='cot{}',
        xlabel='Drive',
        equation=equation_cot,
        ylabel='Cost of transport ({equation})',
    )

    # Velocity / Cost of transport
    conditional_plot(
        conditions=conditions,
        function=plot_multi_exponent,
        plots=plots,
        exp_data=exp_data,
        plot_name='velcot{}',
        xdata='average_velocity',
        ydata='cot{}',
        xlabel='Velocity [m/s]',
        equation=equation_cot,
        ylabel='Cost of transport ({equation})',
        marker='o',
        linestyle='',
    )

    # Velocity / Torque integral
    conditional_plot(
        conditions=conditions,
        function=plot_multi_exponent,
        plots=plots,
        exp_data=exp_data,
        plot_name='veltrq{}',
        xdata='average_velocity',
        ydata='torque_integral{}',
        xlabel='Velocity [m/s]',
        equation=equation_torque_integral,
        ylabel='{equation}',
        marker='o',
        linestyle='',
    )

    # Trajectories
    conditional_plot(
        conditions=conditions,
        function=plot_trajectories,
        plots=plots,
        exp_data=exp_data,
        plot_name='trajectories',
    )

    # Standing
    conditional_plot(
        conditions=conditions,
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='standing',
        xdata='drive',
        ydata='gait_Stand',
        xlabel='Drive',
        ylabel='Standing factor',
        ylim=[0, 1],
    )

    # Duty factor
    conditional_plot(
        conditions=conditions,
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='duty_factor',
        xdata='drive',
        ydata='gait_DF',
        xlabel='Drive',
        ylabel='Standing factor',
        ylim=[0, 1],
    )

    if list(exp_data.values())[0][0]['n_legs'] == 4:

        # Gait
        for gait_i, gait in enumerate(['Trotting', 'Sequence', 'Bound']):
            label_template = (
                f'{{label}} - {gait}'
                if len(exp_data) > 1
                else gait
            )
            conditional_plot(
                conditions=conditions,
                function=plot_element,
                plots=plots,
                exp_data=exp_data,
                plot_name='gait_scores',
                xdata='drive',
                ydata=f'gait_{gait}',
                xlabel='Drive',
                ylabel='Score',
                label_template=label_template,
                zorder_i=gait_i,
                ylim=[0, 1],
                show_legend=True,
            )

        # Duty cycles
        for key_i, key in enumerate(['LF', 'RF', 'LH', 'RH']):
            label_template = (
                f'{{label}} - {key}'
                if len(exp_data) > 1
                else '{{label}}'
            )
            conditional_plot(
                conditions=conditions,
                function=plot_element,
                plots=plots,
                exp_data=exp_data,
                plot_name='duty_factors',
                xdata='drive',
                ydata=f'gait_{gait}',
                xlabel='Drive',
                ylabel='Score',
                label_template=label_template,
                zorder_i=key_i,
                ylim=[0, 1],
                show_legend=True,
            )


def main():
    """Main"""

    # Clargs
    clargs = parse_args()
    assert len(clargs.logs) == len(clargs.labels), (
        f'{len(clargs.logs)=} != {len(clargs.labels)=}'
    )

    # Data obtained for plotting
    exp_data = load_data(
        sweep_type=clargs.type,
        logs=zip(clargs.logs, clargs.labels),
    )

    # Plot figure
    plots = {}
    if clargs.type == 'drive':
        plot_drive(plots=plots, exp_data=exp_data)

    # Save plots
    extension = clargs.extension
    for name, fig in plots.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    profile(main)
