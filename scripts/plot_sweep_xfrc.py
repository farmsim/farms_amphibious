# """Plot sweep"""

# import os

# import numpy as np

# from farms_core import pylog
# from farms_core.utils.profile import profile
# from farms_core.simulation.options import SimulationOptions
# from farms_core.analysis.metrics import com_velocities_norm
# from farms_core.analysis.plot import plt_farms_style

# from farms_amphibious.data.data import AmphibiousData
# from farms_amphibious.model.options import AmphibiousOptions
# from farms_amphibious.analysis.plot import plot_element_colorgraph
# from farms_amphibious.utils.parse_args import parse_args_sweep


# def load_experiment(_sweep_type, exp_data, log, name, label):
#     """Load experiment"""

#     # Load
#     animat_options_path = os.path.join(log, 'animat_options.yaml')
#     animat_options = AmphibiousOptions.load(animat_options_path)
#     data_path = os.path.join(log, 'simulation.hdf5')
#     animat_data = AmphibiousData.from_file(data_path)
#     timestep = animat_data.timestep
#     simulation_options_path = os.path.join(log, 'simulation_options.yaml')
#     simulation_options = SimulationOptions.load(simulation_options_path)

#     # Xfrc
#     data_xfrc =  animat_data.sensors.xfrc
#     sampling = 10  # [Hz]
#     xfrc = np.asarray(data_xfrc.forces())[
#         np.arange(
#             start=0,
#             stop=simulation_options.n_iterations-2,  # n_iterations-1,
#             step=max(1, round(1/(sampling*timestep))),
#             dtype=int,
#         )
#     ]

#     # Data
#     if label not in exp_data:
#         exp_data[label] = []
#     exp_data[label].append({
#         'name': name,
#         'label': label,
#         'drive': animat_options.control.network.drives[0].initial_value,
#         'xfrc_x': xfrc[:, :, 0],
#         'xfrc_y': xfrc[:, :, 1],
#         'xfrc_z': xfrc[:, :, 2],
#         'duration': simulation_options.duration(),
#     })


# def load_data(sweep_type, logs):
#     """Load data"""
#     exp_data = {}
#     for log, name, label in logs:
#         load_experiment(sweep_type, exp_data, log, name, label)
#     return exp_data


# def conditional_plot(conditions, function, plot_name, **kwargs):
#     """Conditional plot"""
#     for condition in conditions:
#         function(
#             plot_name=plot_name+condition['suffix'],
#             condition=condition['condition'],
#             **kwargs,
#         )


# def plot_motion(plots, exp_data):
#     """Plot motion"""

#     # Conditions
#     conditions = [{'suffix': '', 'condition': lambda _: True}]

#     # Xfrc
#     duration = list(exp_data.values())[0][0]['duration']
#     for direction in ['x', 'y', 'z']:
#         conditional_plot(
#             conditions=conditions,
#             function=plot_element_colorgraph,
#             plots=plots,
#             exp_data=exp_data,
#             plot_name=f'xfrc_{direction}',
#             xdata=f'xfrc_{direction}',
#             xlabel='Time [s]',
#             x_extent=[0, duration],
#             ydata='name',
#             ylabel='Experiments',
#             clabel='Force [N]',
#             cmap='cividis',
#             aspect=3.0,
#             n_pixel_y=4,
#             gap=1,
#         )


# def main():
#     """Main"""

#     # Matplolib options
#     plt_farms_style()

#     # Clargs
#     clargs = parse_args_sweep()

#     # Data obtained for plotting
#     exp_data = load_data(
#         sweep_type=clargs.type,
#         logs=zip(clargs.logs, clargs.names, clargs.labels),
#     )

#     # Plot figure
#     plots = {}
#     plot_motion(plots=plots, exp_data=exp_data)

#     # Save plots
#     extension = clargs.extension
#     for name, fig in plots.items():
#         filename = os.path.join(clargs.output, f'{name}.{extension}')
#         pylog.debug('Saving to %s', filename)
#         fig.savefig(filename, format=extension, bbox_inches='tight', dpi=300)


# if __name__ == '__main__':
#     profile(main)
