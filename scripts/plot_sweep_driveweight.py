"""Plot sweep"""

import os

from farms_core import pylog
from farms_core.io.yaml import yaml2pyobject
from farms_core.utils.profile import profile
from farms_core.analysis.plot import plt_farms_style

from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.parse_args import parse_args_sweep
from farms_amphibious.analysis.plot import (
    plot_element_2d,
    plot_multi_exponent_2d,
)


def get_weight_osc(animat_options):
    """Get oscillator coupling weight"""
    if not animat_options.control.network.osc2osc:
        return 0
    index = 0
    return animat_options.control.network.osc2osc[index]['weight']


def get_weight_stretch(animat_options):
    """Get stretch feedback weight"""
    if not animat_options.control.network.joint2osc:
        return 0
    index = 0
    return abs(animat_options.control.network.joint2osc[index]['weight'])


def get_weight_xfrc(animat_options):
    """Get xfrc feedback weight"""
    if not animat_options.control.network.xfrc2osc:
        return 0
    index = 0
    return animat_options.control.network.xfrc2osc[index]['weight']


def load_experiment(sweep_type, exp_data, log, label):
    """Load experiment"""

    # Load
    animat_options_path = os.path.join(log, 'animat_options.yaml')
    animat_options = AmphibiousOptions.load(animat_options_path)
    analysis = yaml2pyobject(os.path.join(log, 'analysis.yaml'))

    # Get drive
    drive = animat_options.control.network.drives[0].initial_value

    # Get weight
    weight = {
        'drivecoupling': get_weight_osc,
        'drivestretch': get_weight_stretch,
        'drivexfrc': get_weight_xfrc,
        'drivexfrccos': get_weight_xfrc,
    }[sweep_type](animat_options)

    # Torques integral
    torque_integrals = [
        analysis['metrics'][f'torque_integral{exponent}']
        for exponent in range(1, 5)
    ]

    # Average velocity
    average_velocity = analysis['metrics']['velocity']

    # Data
    if label not in exp_data:
        exp_data[label] = []
    metrics = {
        'drive': drive,
        'weight': weight,
        'average_velocity': average_velocity,
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
    for key, value in analysis['metrics']['frequency'].items():
        metrics[f'frequency_{key}'] = value
    metrics['cot'] = analysis['metrics']['cot']
    exp_data[label].append(metrics)

    # Gaits
    for key, value in analysis['metrics']['gait'].items():
        exp_data[label][-1][f'gait_{key}'] = value


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


def plot_driveweight(plots, exp_data, sweep_type):
    """Plot drive"""

    # Conditions
    conditions = [{'suffix': '', 'condition': lambda _: True}]

    # Maps
    label_weight = {
        'drivecoupling': 'Coupling weight',
        'drivestretch': 'Stretch feedback weight',
        'drivexfrc': 'Hydrodynamics feedback weight',
        'drivexfrccos': 'Hydrodynamics feedback weight',
    }[sweep_type]
    identifier_weight = {
        'drivecoupling': 'coupling',
        'drivestretch': 'stretch',
        'drivexfrc': 'xfrc',
        'drivexfrccos': 'xfrccos',
    }[sweep_type]

    # Velocities
    conditional_plot(
        conditions=conditions,
        function=plot_element_2d,
        plots=plots,
        exp_data=exp_data,
        plot_name=f'drv_{identifier_weight}_vel',
        xdata='drive',
        ydata='weight',
        zdata='average_velocity',
        xlabel='Drive',
        ylabel=label_weight,
        zlabel='Velocity [m/s]',
    )

    # Frequencies
    conditional_plot(
        conditions=conditions,
        function=plot_element_2d,
        plots=plots,
        exp_data=exp_data,
        plot_name=f'drv_{identifier_weight}_frq',
        xdata='drive',
        ydata='weight',
        zdata='frequency_all',
        xlabel='Drive',
        ylabel=label_weight,
        zlabel='Frequency [Hz]',
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
        function=plot_multi_exponent_2d,
        plots=plots,
        exp_data=exp_data,
        plot_name=f'drv_{identifier_weight}_torque_integral{{}}',
        xdata='drive',
        ydata='weight',
        zdata='torque_integral{}',
        xlabel='Drive',
        ylabel=label_weight,
        zlabel='{equation}',
        equation=equation_torque_integral,
        log=True,
    )

    # Cost of transport
    conditional_plot(
        conditions=conditions,
        function=plot_element_2d,
        plots=plots,
        exp_data=exp_data,
        plot_name=f'drv_{identifier_weight}_cot_2d',
        xdata='drive',
        ydata='weight',
        zdata='cot',
        xlabel='Drive',
        ylabel=label_weight,
        zlabel=f'CoT: {equation_cot}',
        log=True,
    )
    conditional_plot(
        conditions=conditions,
        function=plot_multi_exponent_2d,
        plots=plots,
        exp_data=exp_data,
        plot_name=f'drv_{identifier_weight}_cot{{}}_2d',
        xdata='drive',
        ydata='weight',
        zdata='cot{}',
        xlabel='Drive',
        ylabel=label_weight,
        zlabel='CoT: {equation}',
        equation=equation_cot2,
        log=True,
    )

    # Velocity / Frequency
    conditional_plot(
        conditions=conditions,
        function=plot_element_2d,
        plots=plots,
        exp_data=exp_data,
        plot_name=f'vel_{identifier_weight}_frq_2d',
        xdata='average_velocity',
        ydata='weight',
        zdata='frequency_all',
        xlabel='Velocity [m/s]',
        ylabel=label_weight,
        zlabel='Frequency [Hz]',
    )

    # Velocity / Cost of transport
    conditional_plot(
        conditions=conditions,
        function=plot_element_2d,
        plots=plots,
        exp_data=exp_data,
        plot_name=f'vel_{identifier_weight}_cot_2d',
        xdata='average_velocity',
        ydata='weight',
        zdata='cot',
        xlabel='Velocity [m/s]',
        ylabel=label_weight,
        zlabel=f'CoT: {equation_cot}',
        log=True,
    )
    conditional_plot(
        conditions=conditions,
        function=plot_multi_exponent_2d,
        plots=plots,
        exp_data=exp_data,
        plot_name=f'vel_{identifier_weight}_cot{{}}',
        xdata='average_velocity',
        ydata='weight',
        zdata='cot{}',
        xlabel='Velocity [m/s]',
        ylabel=label_weight,
        zlabel='CoT: {equation}',
        equation=equation_cot2,
        log=True,
    )

    # Velocity / Torque integral
    conditional_plot(
        conditions=conditions,
        function=plot_multi_exponent_2d,
        plots=plots,
        exp_data=exp_data,
        plot_name=f'vel_{identifier_weight}_trq{{}}',
        xdata='average_velocity',
        ydata='weight',
        zdata='torque_integral{}',
        xlabel='Velocity [m/s]',
        ylabel=label_weight,
        zlabel='{equation}',
        equation=equation_torque_integral,
        log=True,
    )


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
    plot_driveweight(plots=plots, exp_data=exp_data, sweep_type=clargs.type)

    # Save plots
    extension = clargs.extension
    for name, fig in plots.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    profile(main)
