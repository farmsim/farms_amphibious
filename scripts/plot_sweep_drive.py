"""Plot sweep"""

import os

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.analysis.plot import plt_farms_style

from farms_amphibious.analysis.sweep import load_data_drive
from farms_amphibious.analysis.plot import plot_element, plot_multi_exponent
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
        },
    ]

    # Velocities
    conditional_plot(
        conditions=conditions,
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='drv_vel',
        xdata='drive',
        ydata='average_velocity',
        xlabel='Drive',
        ylabel='Velocity [m/s]',
    )
    conditional_plot(
        conditions=conditions,
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='drv_vel_bl',
        xdata='drive',
        ydata='average_velocity_bl',
        xlabel='Drive',
        ylabel='Velocity [BL/s]',
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
        r' \frac{1}{mg\bar{v}}'
        r' \frac{1}{T_1 - T_0}'
        r' \int_{T_0}^{T_1}'
        r' \sum_{j=1}^N \tau_j(t) \omega_j(t)'
        r' dt$'
    )
    equation_cot2 = (
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
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='cot',
        xdata='drive',
        ydata='cot',
        xlabel='Drive',
        ylabel=f'CoT: {equation_cot}',
    )
    conditional_plot(
        conditions=conditions,
        function=plot_multi_exponent,
        plots=plots,
        exp_data=exp_data,
        plot_name='cot{}',
        xdata='drive',
        ydata='cot{}',
        xlabel='Drive',
        equation=equation_cot2,
        ylabel='CoT: {equation}',
    )

    # Velocity / Cost of transport
    conditional_plot(
        conditions=conditions,
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='velcot',
        xdata='average_velocity',
        ydata='cot',
        xlabel='Velocity [m/s]',
        ylabel=f'CoT: {equation_cot}',
    )
    conditional_plot(
        conditions=conditions,
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='velblcot',
        xdata='average_velocity_bl',
        ydata='cot',
        xlabel='Velocity [BL/s]',
        ylabel=f'CoT: {equation_cot}',
    )
    conditional_plot(
        conditions=conditions,
        function=plot_multi_exponent,
        plots=plots,
        exp_data=exp_data,
        plot_name='velcot{}',
        xdata='average_velocity',
        ydata='cot{}',
        xlabel='Velocity [m/s]',
        equation=equation_cot2,
        ylabel='CoT: {equation}',
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

    # Velocity / Frequency
    conditional_plot(
        conditions=conditions,
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='frq_vel',
        xdata='frequency_body',
        ydata='average_velocity',
        xlabel='Frequency [Hz]',
        ylabel='Velocity [m/s]',
    )
    conditional_plot(
        conditions=conditions,
        function=plot_element,
        plots=plots,
        exp_data=exp_data,
        plot_name='frq_vel_bl',
        xdata='frequency_body',
        ydata='average_velocity_bl',
        xlabel='Frequency [Hz]',
        ylabel='Velocity [BL/s]',
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
