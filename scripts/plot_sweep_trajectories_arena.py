"""Plot sweep"""

import os
from functools import partial

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.io.sdf import ModelSDF, Cylinder, Heightmap
from farms_core.simulation.options import SimulationOptions
from farms_core.analysis.plot import plt_farms_style
from farms_core.analysis.metrics import com_positions

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.analysis.plot import plot_trajectories
from farms_amphibious.utils.parse_args import parser_sweep, validate_sweep_clargs
from farms_amphibious.model.options import (
    AmphibiousOptions,
    AmphibiousArenaOptions,
)



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


def plot_arena(arena_options: AmphibiousArenaOptions, model_sdf: ModelSDF):
    """Plot arena"""
    axis = plt.gca()
    arena_pos = np.array(arena_options.spawn.pose[:3])
    for link in model_sdf.links:
        for element in link.collisions:

            # Cylinder
            if isinstance(element.geometry, Cylinder):
                circle = plt.Circle(
                    xy=arena_pos[:2]+element.pose[:2],
                    radius=element.geometry.radius,
                    color='black',
                )
                axis.add_patch(circle)

            # Heightmap
            elif isinstance(element.geometry, Heightmap):
                path = os.path.join(model_sdf.directory, element.geometry.uri)
                assert os.path.isfile(path), path
                img = imread(path)  # Read PNG image
                img = img[:, :, 0] if img.ndim == 3 else img[:, :]  # RGB vs Grey
                vmin, vmax = (np.iinfo(img.dtype).min, np.iinfo(img.dtype).max)
                img = (img - vmin)/(vmax-vmin)  # Normalize
                img = np.flip(img, axis=0)  # Cartesian coordinates
                size = np.array(element.geometry.size[:3])
                pos = np.array(arena_pos[:3]) + np.array(element.pose[:3])
                img = size[2]*(img - 0.5) + pos[2]
                imgplot = plt.imshow(
                    img,
                    extent=(
                        pos[0] - 0.5*size[0],
                        pos[0] + 0.5*size[0],
                        pos[1] - 0.5*size[1],
                        pos[1] + 0.5*size[1],
                    ),
                    aspect='auto',
                    origin='lower',
                )
                cmap = 'cividis'
                imgplot.set_cmap(cmap)
                divider = make_axes_locatable(axis)
                cax = divider.append_axes('top', size='5%', pad=0.5)
                cbar = plt.colorbar(imgplot, cax=cax, orientation='horizontal')
                # cbar.ax.xaxis.set_ticks_position('top')
                cbar.set_label('Ground height [m]')


def plot_motion(plots, exp_data, arena_config):
    """Plot motion"""

    # Conditions
    conditions = [{'suffix': '', 'condition': lambda _: True}]

    # Arena options
    arena_options = AmphibiousArenaOptions.load(arena_config)
    model_sdf = ModelSDF.read(filename=os.path.expandvars(arena_options.sdf))[0]

    # Trajectories
    plot_function = partial(
        conditional_plot,
        conditions=conditions,
        function=plot_trajectories,
        plots=plots,
        exp_data=exp_data,
    )
    plot_function(plot_name='trajectories_arena', legend=True)
    plot_arena(arena_options, model_sdf)
    plot_function(plot_name='trajectories_arena_no_legend', legend=False)
    plot_arena(arena_options, model_sdf)


def main():
    """Main"""

    # Matplolib options
    plt_farms_style()

    # Clargs
    parser = parser_sweep()
    parser.add_argument(
        '--arena_config',
        type=str,
        help='Arena config',
    )
    clargs = parser.parse_args()
    validate_sweep_clargs(clargs)
    assert clargs.arena_config, f'{clargs.arena_config=}'

    # Data obtained for plotting
    exp_data = load_data(
        sweep_type=clargs.type,
        logs=zip(clargs.logs, clargs.labels),
    )

    # Plot figure
    plots = {}
    plot_motion(
        plots=plots,
        exp_data=exp_data,
        arena_config=clargs.arena_config,
    )

    # Save plots
    extension = clargs.extension
    for name, fig in plots.items():
        filename = os.path.join(clargs.output, f'{name}.{extension}')
        pylog.debug('Saving to %s', filename)
        fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    profile(main)
