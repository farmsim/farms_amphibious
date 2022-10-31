"""Plot joints"""

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
    clargs = parse_args_postprocessing(description='Plot amphibious joints')

    # Load data
    animat_options = AmphibiousOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AmphibiousData.from_file(clargs.data)
    n_iterations = simulation_options.n_iterations

    # Plot simulation data
    times = simulation_options.times()
    assert len(times) == n_iterations, f'{len(times)=} != {n_iterations=}'
    times = times[:animat_data.sensors.links.array.shape[0]]

    # Plot joints positions
    joints_pos = np.array(animat_data.sensors.joints.positions_all())
    joints_vel = np.array(animat_data.sensors.joints.velocities_all())
    joints_mtrq = np.array(animat_data.sensors.joints.motor_torques_all())
    joints_ctrq = np.array(animat_data.sensors.joints.cmd_torques())
    joints_atrq = np.array(animat_data.sensors.joints.active_torques())
    joints_strq = np.array(animat_data.sensors.joints.spring_torques())
    joints_dtrq = np.array(animat_data.sensors.joints.damping_torques())
    joints_ftrq = np.array(animat_data.sensors.joints.friction_torques())
    labels = animat_data.sensors.joints.names

    # Convention
    convention = AmphibiousConvention.from_amphibious_options(animat_options)
    n_joints = len(labels)
    all_indices = list(range(n_joints))
    body_indices = list(range(convention.n_joints_body))
    legs_indices = list(range(
        convention.n_joints_body,
        convention.n_joints_body+convention.n_joints_legs(),
    ))

    # Plot
    plots_sim = {}
    for data, clabel, suffix in [
            [joints_pos, 'Position [rad]', '_positions'],
            [joints_vel, 'Velocitiy [rad/s]', '_velocities'],
            [joints_ctrq, 'Command torque [Nm]', '_command_torque'],
            [joints_mtrq, 'Motor torque [Nm]', '_motor_torque'],
            [joints_atrq, 'Active torque [Nm]', '_active_torque'],
            [joints_strq, 'Spring torque [Nm]', '_spring_torque'],
            [joints_dtrq, 'Damping torque [Nm]', '_damping_torque'],
            [joints_ftrq, 'Friction torque [Nm]', '_friction_torque'],
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
                    ['_legs', legs_indices, 1.0],
            ]:
                fig = plt.figure(f'colorgraph_joints{suffix}{suffix2}{suffix3}')
                plots_sim[f'colorgraph_joints{suffix}{suffix2}{suffix3}'] = fig
                if not indices:
                    continue
                colorgraph(
                    data=data.T[indices, :],
                    labels=np.array(labels)[indices],
                    n_pixel_y=4,
                    gap=1,
                    vmin=np.percentile(data.flatten(), 1),
                    vmax=np.percentile(data.flatten(), 99),
                    x_extent=[0, times[-1]],
                    cmap=cmap,
                    xlabel='Time [s]',
                    ylabel='Joints',
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
