"""Plot sweep"""

import os

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.analysis.plot import plt_farms_style

from farms_amphibious.analysis.plot import plot_element
from farms_amphibious.analysis.sweep import load_data_drive
from farms_amphibious.utils.parse_args import parse_args_sweep


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

    # Conditions
    conditions = [{'suffix': '', 'condition': lambda _: True}]

    # Gait
    for gait_i, gait in enumerate(['Trotting', 'Sequence']):  # , 'Bound'
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
            else key
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

    # Matplolib options
    plt_farms_style()

    # Clargs
    clargs = parse_args_sweep()

    # Data obtained for plotting
    exp_data = load_data_drive(logs=zip(clargs.logs, clargs.labels))

    # Plot figure
    plots = {}
    plot_drive(plots=plots, exp_data=exp_data)

    # Save plots
    extension = clargs.extension
    for name, fig in plots.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    profile(main)
