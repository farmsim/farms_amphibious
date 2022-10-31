"""Plot gait"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, ArrowStyle, ConnectionStyle
from scipy.interpolate import interp1d
from PyPDF2 import PdfFileReader
from farms_core.model.data import AnimatData
from farms_core.analysis.plot import plt_farms_style
from farms_core.simulation.options import SimulationOptions
from farms_core.io.yaml import yaml2pyobject, pyobject2yaml
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.model.convention import AmphibiousConvention


def argument_parser() -> ArgumentParser:
    """Argument parser"""
    parser = ArgumentParser(
        description='FARMS gait plotting',
        formatter_class=(
            lambda prog:
            ArgumentDefaultsHelpFormatter(prog, max_help_position=50)
        ),
    )
    parser.add_argument(
        '--plot_type',
        type=str,
        choices=('gait', 'path'),
        default=None,
        help='Plot type',
    )
    parser.add_argument(
        '--sim_data',
        type=str,
        help='Simulation data path',
    )
    parser.add_argument(
        '--sim_config',
        type=str,
        help='Simulation data path',
    )
    parser.add_argument(
        '--animat_config',
        type=str,
        help='Animat config path',
    )
    parser.add_argument(
        '--snapshots_render',
        type=str,
        help='Snapshots render path',
    )
    parser.add_argument(
        '--snapshots_config',
        type=str,
        help='Snapshots config path',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path',
    )
    parser.add_argument(
        '--output_config',
        type=str,
        help='Output config path',
    )
    parser.add_argument(
        '--figsize',
        metavar='size_x, size_y',
        type=float,
        nargs=2,
        default=(7, 10),
        help='Figure size',
    )
    parser.add_argument(
        '--dpi',
        type=float,
        default=600,
        help='Output path',
    )
    parser.add_argument(
        '--use_links',
        action='store_true',
        help='Use links',
    )
    return parser


def parse_args() -> Namespace:
    """Parse arguments"""
    parser = argument_parser()
    return parser.parse_args()


def transform(point, mov, rot):
    """Transform"""
    return np.dot(rot, point+mov)


def snapshot_links_positions(
        snapshot_i, iteration,
        links_sensors, indices,
        sep, mov, rot,
        use_links=False,
):
    """Snapshot links positions"""
    position_function = (
        links_sensors.urdf_position
        if use_links
        else links_sensors.com_position
    )
    pos_local = [
        transform(
            point=position_function(iteration=iteration, link_i=link_i)[:2],
            mov=mov,
            rot=rot,
        )
        for link_i in indices
    ]
    return np.array([[pos[0], pos[1] + sep*snapshot_i] for pos in pos_local])


def plot_snapshot_links_positions(interpolate=False, **kwargs):
    """Plot snapshot links positions"""
    style = kwargs.pop('style', 'ko-')
    alpha = kwargs.pop('alpha', 0.5)
    markersize = kwargs.pop('markersize', 1)
    linewidth = kwargs.pop('linewidth', 1)
    label = kwargs.pop('label', None)
    pos_plot = snapshot_links_positions(**kwargs)
    if interpolate:
        n_points = pos_plot.shape[0]
        f_data_x = interp1d(range(n_points), pos_plot[:, 0], kind='cubic')
        f_data_y = interp1d(range(n_points), pos_plot[:, 1], kind='cubic')
        data_x = f_data_x(np.linspace(0, n_points-1, 20))
        data_y = f_data_y(np.linspace(0, n_points-1, 20))
    else:
        data_x, data_y = pos_plot[:, 0], pos_plot[:, 1]
    plt.plot(
        data_x, data_y,
        style, alpha=alpha, markersize=markersize, linewidth=linewidth,
        label=label,
    )
    return pos_plot


def main():
    """Main"""

    # Style
    plt_farms_style()

    # Command line arguments
    clargs = parse_args()

    # Aquire data
    snapshots_config = yaml2pyobject(clargs.snapshots_config)
    img = mpimg.imread(clargs.snapshots_render)
    data = AnimatData.from_file(clargs.sim_data)
    links_sensors = data.sensors.links
    contacts_sensors = data.sensors.contacts
    sep = np.linalg.norm(snapshots_config['separation'])
    iterations = snapshots_config['iterations']
    n_snapshots = len(iterations)
    sim_options = SimulationOptions.load(clargs.sim_config)
    convention = AmphibiousConvention.from_amphibious_options(
        animat_options=AmphibiousOptions.load(clargs.animat_config),
    )

    # Frame
    frame_pose = snapshots_config['frame_pos']
    origin_pose = snapshots_config['origin_pos']
    frame_x = snapshots_config['frame_x']
    frame_y = snapshots_config['frame_y']
    if clargs.plot_type == 'gait':
        mov = -np.array(frame_pose)
        rot = np.linalg.inv(np.array([frame_x, frame_y]).T)
    else:
        mov = -np.array(origin_pose)
        rot = np.array([frame_x, -np.array(frame_y)])
    camera_dimensions = snapshots_config['bounds_diff']

    # Plot figure
    _fig, axes = plt.subplots(1, 1, figsize=clargs.figsize)

    # Show image
    extent = (
        [0, camera_dimensions[0], camera_dimensions[1], 0]
        if clargs.plot_type == 'gait'
        else [0, camera_dimensions[0], 0, camera_dimensions[1]]
    )
    _imgplot = plt.imshow(X=img, extent=extent, origin='upper')

    # CoM
    com_global = np.array([
        links_sensors.global_com_position(iteration=iteration)[:2]
        for iteration in iterations
    ])
    com_camera = [transform(point=pos, mov=mov, rot=rot) for pos in com_global]
    com_camera = np.array(com_camera)
    com_mean = np.mean(com_camera[:, 1])

    # Plot for each snapshot
    head_pos_local = []
    has_foot = 'foot_0_0' in links_sensors.names
    body_mark_options={
        'alpha': 0.5,
        'linewidth': 0.7,
        'markersize': 0.5,
    }
    com_pos_plot = [[], []]
    contacts_plots = [[], []]
    links_label = False
    for i, (iteration, com_pos) in enumerate(zip(iterations, com_camera)):

        # CoM positions for plot
        com_pos_plot[0].append(com_pos[0])
        com_pos_plot[1].append(com_pos[1]+sep*i)

        # Plot body positions
        pos_plot = plot_snapshot_links_positions(
            interpolate=True,
            snapshot_i=i, iteration=iteration, links_sensors=links_sensors,
            indices=range(convention.n_links_body()),
            sep=sep, mov=mov, rot=rot,
            use_links=clargs.use_links,
            **body_mark_options,
            **({'label': 'Links positions'} if links_label else {}),
        )
        head_pos_local.append(pos_plot[0])
        if links_label:
            links_label = False

        # Limbs analysis
        for leg_i in range(convention.n_legs_pair()):
            for side_i in range(2):

                # Plot limbs positions
                pos_plot = plot_snapshot_links_positions(
                    snapshot_i=i, iteration=iteration,
                    links_sensors=links_sensors,
                    indices=(
                        [  # Limb
                            convention.leglink2index(leg_i, side_i, joint_i)
                            for joint_i in range(convention.n_dof_legs)
                        ]
                    ),
                    sep=sep, mov=mov, rot=rot,
                    use_links=clargs.use_links,
                    **body_mark_options,
                )

                # Plot contacts
                if has_foot:
                    force = np.linalg.norm(contacts_sensors.total(
                        iteration=iteration,
                        sensor_i=2*leg_i+side_i,
                    ))
                    if force > 1e-3:
                        contacts_plots[0].append(pos_plot[-1, 0])
                        contacts_plots[1].append(pos_plot[-1, 1])

    # Plot CoM
    plt.plot(
        com_pos_plot[0], com_pos_plot[1], 'r*', alpha=.7,
        label='CoM position',
    )
    contacts_plot = plt.plot(
        contacts_plots[0], contacts_plots[1],
        'C1o', markersize=3, alpha=0.5,
        label='Contacts',
    )
    for line in contacts_plot:  # Background
        line.set_zorder(0)

    # Final layout
    plt.xlabel('Distance [m]')
    plt.grid(visible=True, alpha=0.5)

    if clargs.plot_type == 'gait':

        # Axis
        axes.xaxis.grid(visible=False)
        axes.set_axisbelow(True)
        for label in ['top', 'right', 'left']:  # 'bottom',
            axes.spines[label].set_visible(False)

        # Head advancement
        head_pos_local = np.array(head_pos_local)
        plt.plot(
            head_pos_local[[0, -1], 0], head_pos_local[[0, -1], 1],
            'k--', alpha=.3, linewidth=0.5,
        )

        # Snapshots ticks
        yticks = [sep*i+com_mean for i in range(n_snapshots)]
        plt.yticks(ticks=yticks, labels=range(1, n_snapshots+1))
        plt.ylim([yticks[-1] + sep, yticks[0] - sep])
        # plt.ylim([yticks[-1] + sep, yticks[0] - sep])

        # Time
        sim_options = SimulationOptions.load(clargs.sim_config)
        time_interval = (iterations[1] - iterations[0])*sim_options.timestep
        rel_pos = 0.5
        plt.text(
            x=0.1*sep,
            y=(1-rel_pos)*yticks[-2]+rel_pos*yticks[-1],
            s=f'{round(1e3*time_interval)} [ms]',
            va='center',
            ha='left',
            fontsize=8,
            color='k',
            animated=False,
        )
        arrow = FancyArrowPatch(
            posA=[0.15*sep, yticks[-2]],
            posB=[0.15*sep, yticks[-1]],
            arrowstyle=ArrowStyle(
                stylename='Fancy',
                head_length=10,
                head_width=5,
                tail_width=1,
            ),
            connectionstyle=ConnectionStyle('Arc3', rad=0.2),
            color='k',
        )
        axes.add_artist(arrow)

    elif clargs.plot_type == 'path':

        # Bounds and label
        plt.xlabel('Position [m]')
        plt.ylabel('Position [m]')

        # CoM trajectory
        com_global_all = np.array([
            links_sensors.global_com_position(iteration=iteration)[:2]
            for iteration in range(iterations[0], iterations[-1])
        ])
        com_camera_all = [
            transform(point=pos, mov=mov, rot=rot)
            for pos in com_global_all
        ]
        com_camera_all = np.array(com_camera_all)
        plt.plot(
            com_camera_all[:, 0], com_camera_all[:, 1], 'r--', alpha=0.5,
            label='CoM trajectory',
        )
        axes.xaxis.grid(visible=True)

    else:
        raise Exception(f'Unknown plot type: {clargs.plot_type}')

    # Legend
    if clargs.plot_type != 'gait':
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save figure
    plt.savefig(clargs.output, dpi=clargs.dpi, bbox_inches='tight')

    # Get figure info
    time_total = (iterations[-1] - iterations[0])*sim_options.timestep
    velocity = np.linalg.norm(com_global[-1] - com_global[0])/time_total
    with open(clargs.output, 'rb') as pdf_file:
        pdf = PdfFileReader(pdf_file)
        res = [float(val) for val in pdf.getPage(0).mediaBox]
    pyobject2yaml(clargs.output_config, {
        'velocity': float(velocity),
        'box': res,
    })


if __name__ == '__main__':
    main()
