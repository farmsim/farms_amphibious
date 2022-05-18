"""Plot gait"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, ArrowStyle, ConnectionStyle
from PyPDF2 import PdfFileReader
from farms_core.io.yaml import yaml2pyobject, pyobject2yaml
from farms_core.simulation.options import SimulationOptions
from farms_core.model.data import AnimatData
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


def plot_snapshot_links_positions(**kwargs):
    """Plot snapshot links positions"""
    style = kwargs.pop('style', 'ko-')
    alpha = kwargs.pop('alpha', 0.5)
    markersize = kwargs.pop('markersize', 1)
    linewidth = kwargs.pop('linewidth', 1)
    pos_plot = snapshot_links_positions(**kwargs)
    plt.plot(
        pos_plot[:, 0], pos_plot[:, 1],
        style, alpha=alpha, markersize=markersize, linewidth=linewidth,
    )
    return pos_plot


def main():
    """Main"""
    clargs = parse_args()

    # LaTeX
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Aquire data
    camera = yaml2pyobject(clargs.snapshots_config)
    img = mpimg.imread(clargs.snapshots_render)
    data = AnimatData.from_file(clargs.sim_data)
    links_sensors = data.sensors.links
    contacts_sensors = data.sensors.contacts
    sep = np.linalg.norm(camera['separation'])
    iterations = camera['iterations']
    n_snapshots = len(iterations)
    convention = AmphibiousConvention.from_amphibious_options(
        animat_options=AmphibiousOptions.load(clargs.animat_config),
    )

    # Frame
    frame_pose = camera['frame_pos']
    frame_x = camera['frame_x']
    frame_y = camera['frame_y']
    mov = -np.array(frame_pose)
    rot = np.linalg.inv(np.array([frame_x, frame_y]).T)
    camera_dimensions = camera['bounds_diff']

    # Plot figure
    _fig, axes = plt.subplots(1, 1, figsize=clargs.figsize)

    # Show image
    _imgplot = plt.imshow(
        X=img,
        extent=[0, camera_dimensions[0], camera_dimensions[1], 0],
    )

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
    for i, (iteration, pos) in enumerate(zip(iterations, com_camera)):

        # Plot body positions
        pos_plot = plot_snapshot_links_positions(
            snapshot_i=i, iteration=iteration, links_sensors=links_sensors,
            indices=range(convention.n_links_body()),
            sep=sep, mov=mov, rot=rot,
            use_links=clargs.use_links,
        )
        head_pos_local.append(pos_plot[0])

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
                        + (  # Foot
                            [-convention.n_legs + 2*leg_i + side_i]
                            if has_foot
                            else []
                        )
                    ),
                    sep=sep, mov=mov, rot=rot,
                    use_links=clargs.use_links,
                )

                # Plot contacts
                if has_foot:
                    force = np.linalg.norm(contacts_sensors.total(
                        iteration=iteration,
                        sensor_i=2*leg_i+side_i,
                    ))
                    if force > 1e-3:
                        lines = plt.plot(
                            pos_plot[-1, 0], pos_plot[-1, 1],
                            'C1o', markersize=3,
                        )
                        for line in lines:  # Background
                            line.set_zorder(0)

        # Plot CoM
        plt.plot(pos[0], pos[1]+sep*i, 'r*', alpha=.7)

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

    # Final layout
    plt.xlabel('Distance [m]')
    plt.grid(visible=True, alpha=0.5)
    axes.set_axisbelow(True)
    axes.xaxis.grid(visible=False)
    for label in ['top', 'right', 'left']:  # 'bottom',
        axes.spines[label].set_visible(False)

    # Save figure
    plt.savefig(clargs.output, dpi=clargs.dpi, bbox_inches='tight')

    # Get figure info
    time_total = (iterations[-1] - iterations[0])*sim_options.timestep
    velocity = np.linalg.norm(com_global[-1] - com_global[0])/time_total
    with open(clargs.output, 'rb') as pdf_file:
        pdf = PdfFileReader(pdf_file)
        res = [float(val) for val in pdf.getPage(0).mediaBox]
    pyobject2yaml(clargs.output_config, {
        'velocity': velocity,
        'box': res
    })


if __name__ == '__main__':
    main()
