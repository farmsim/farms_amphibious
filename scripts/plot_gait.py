"""Plot gait"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from farms_core import pylog
from farms_core.sensors.sensor_convention import sc
from farms_core.simulation.options import SimulationOptions
from farms_core.analysis.plot import plt_farms_style, colorgraph
from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.model.convention import AmphibiousConvention
from farms_amphibious.utils.parse_args import parse_args_postprocessing


def main():
    """Main"""

    # Style
    plt_farms_style()

    # Clargs
    clargs = parse_args_postprocessing(description='Plot amphibious gait')

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

    convention = AmphibiousConvention.from_amphibious_options(animat_options)
    labels = [tuple(name)[0] for name in animat_data.sensors.contacts.names]
    n_contacts = len(labels)
    all_indices = list(range(n_contacts))
    legs_indices = list(range(convention.n_legs))
    body_indices = list(range(
        convention.n_legs,
        convention.n_legs+convention.n_links_body(),
    ))

    # Plot Contacts
    plots_gait = {}
    swing = np.linalg.norm(
        animat_data.sensors.contacts.array[
            :, :,
            sc.contact_total_x:sc.contact_total_z+1
        ],
        axis=-1,
    )
    polyorder = 3
    window_length = min(int(0.1/timestep//2)*2+1, n_iterations//2*2-1)
    if window_length > polyorder:
        for sensor_i in range(swing.shape[1]):
            non_zeros = np.where(swing[:, sensor_i], 1, 0)
            swing[:, sensor_i] = savgol_filter(
                x=swing[:, sensor_i],
                window_length=window_length,
                polyorder=polyorder,
            )
            swing[:, sensor_i] *= non_zeros
    for cmap, suffix in [
            ['gist_heat_r', ''],
            # ['cividis', '_cividis'],
            # ['Greys', '_greys'],
            # ['turbo', '_turbo'],
            # ['GnBu', '_gnbu'],
            # ['hot_r', '_hotr'],
    ]:
        for suffix2, indices, aspect, feet_only in [
                ['', all_indices, 0.3, False],
                ['_feet', legs_indices, 1.0, True],
                ['_body', body_indices, 1.0, False],
        ]:
            use_limbs = feet_only and convention.n_legs == 4
            labels_plot = (
                ['LF', 'RF', 'LH', 'RH']
                if use_limbs
                else np.array(labels)[indices]
            )
            ylabel = (
                'Limbs'
                if use_limbs
                else 'Contacts sensors'
            )
            fig = plt.figure(f'colorgraph_contacts{suffix}{suffix2}', figsize=(24, 3))
            colorgraph(
                data=swing.T[indices, :],
                labels=labels_plot,
                n_pixel_x=1,
                n_pixel_y=4,
                gap=1,
                vmin=0,
                vmax=1.05*np.percentile(swing.flatten(), 95),
                x_extent=[0, times[-1]],
                cmap=cmap,
                xlabel='Time [s]',
                ylabel=ylabel,
                clabel='Force [N]',
                aspect=aspect,
            )
            plots_gait[f'colorgraph_contacts{suffix}{suffix2}'] = fig

    # Plot gait values
    fig = plt.figure('Gait histogram')
    if animat_options.morphology.n_legs == 4:

        threshold = 1e-16
        contacts_array = np.array(animat_data.sensors.contacts.array)
        swing = np.linalg.norm(
            contacts_array[:, :4, sc.contact_total_x:sc.contact_total_z+1],
            axis=-1,
        ) < threshold  # True if in swing, False otherwise

        indices = [
            convention.legjoint2index(leg_i=leg_i, side_i=side_i, joint_i=0)
            for leg_i in range(2)
            for side_i in range(2)
        ]
        joints_array = np.array(animat_data.sensors.joints.array)
        swing = np.logical_or(
            swing,
            joints_array[:, indices, sc.joint_velocity] > 0,
        )

        gaits = {}
        not_all_ground_indices = np.where(np.sum(swing, axis=1) != 0)[0]
        contacts_gait = swing[not_all_ground_indices, :]
        gaits['Stand'] = np.mean(np.sum(swing, axis=1) == 0)
        gaits['Trotting'] = np.mean(
            (
                np.logical_and(contacts_gait[:, 0], contacts_gait[:, 3])
                + np.logical_and(contacts_gait[:, 1], contacts_gait[:, 2])
            ) == 1,
        )
        gaits['Sequence'] = np.mean(np.logical_or(
            np.sum(contacts_gait, axis=1) == 1,
            np.sum(contacts_gait, axis=1) == 0,
        ))
        gaits['Bound'] = np.mean(
            (
                np.logical_and(contacts_gait[:, 0], contacts_gait[:, 1])
                + np.logical_and(contacts_gait[:, 2], contacts_gait[:, 3])
            ) == 1,
        )
        gaits['LF'] = np.mean(swing[:, 0] == 0)
        gaits['RF'] = np.mean(swing[:, 1] == 0)
        gaits['LH'] = np.mean(swing[:, 2] == 0)
        gaits['RH'] = np.mean(swing[:, 3] == 0)

        fig, axis = plt.subplots()
        width = 0.5
        indices = np.arange(len(gaits))
        plot_gait = axis.bar(indices, gaits.values(), width)
        axis.set_ylim([0, 1])
        axis.set_ylabel('Score')
        axis.set_title('Gait scores')
        axis.set_xticks(indices, labels=gaits.keys())
        axis.legend()
        axis.bar_label(plot_gait, label_type='center')

    plots_gait['gait_histogram'] = fig

    # Save plots
    extension = 'pdf'
    for name, fig in plots_gait.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()
