"""Parse command line arguments"""

import argparse
import numpy as np
from farms_core.simulation.parse_args import (
    config_argument_parser as farms_core_config_argument_parser,
)
from ..model.options import (
    options_kwargs_float_keys,
    options_kwargs_float_list_keys,
    options_kwargs_int_keys,
    options_kwargs_int_list_keys,
    options_kwargs_str_keys,
    options_kwargs_str_list_keys,
    options_kwargs_bool_keys,
    options_kwargs_sph_float_keys,
)


def config_argument_parser() -> argparse.ArgumentParser:
    """Parse args"""
    parser = farms_core_config_argument_parser()
    parser.description = 'Amphibious simulation config generation'
    parser.add_argument(
        '--sdf',
        type=str,
        default='',
        help='SDF file',
    )
    parser.add_argument(
        '--animat',
        type=str,
        default='salamander',
        help='Animat',
    )
    parser.add_argument(
        '--version',
        type=str,
        default='',
        help='Animat version',
    )
    parser.add_argument(
        '--position',
        nargs=3,
        type=float,
        metavar=('x', 'y', 'z'),
        default=(0, 0, 0),
        help='Spawn position',
    )
    parser.add_argument(
        '--orientation',
        nargs=3,
        type=float,
        metavar=('alpha', 'beta', 'gamma'),
        default=(0, 0, 0),
        help='Spawn orientation',
    )
    parser.add_argument(
        '--velocity_lin',
        nargs=3,
        type=float,
        metavar=('vx', 'vy', 'vz'),
        default=(0, 0, 0),
        help='Spawn linear velocity',
    )
    parser.add_argument(
        '--velocity_ang',
        nargs=3,
        type=float,
        metavar=('wx', 'wy', 'wz'),
        default=(0, 0, 0),
        help='Spawn angular velocity',
    )
    parser.add_argument(
        '--arena',
        type=str,
        default='',
        help='Simulation arena',
    )
    parser.add_argument(
        '--arena_sdf',
        type=str,
        default='',
        help='Arena SDF file',
    )
    parser.add_argument(
        '--arena_position',
        nargs=3,
        type=float,
        metavar=('x', 'y', 'z'),
        default=(0, 0, 0),
        help='Arena position',
    )
    parser.add_argument(
        '--arena_orientation',
        nargs=3,
        type=float,
        metavar=('alpha', 'beta', 'gamma'),
        default=(0, 0, 0),
        help='Arena orientation',
    )
    parser.add_argument(
        '--water_height',
        type=float,
        default=None,
        help='Water surface height',
    )
    parser.add_argument(
        '--water_sdf',
        type=str,
        default='',
        help='Water SDF file',
    )
    parser.add_argument(
        '--water_velocity',
        nargs='+',
        type=float,
        default=(0, 0, 0),
        help='Water velocity (For a constant flow, just provide (vx, vy, vz))',
    )
    parser.add_argument(
        '--water_maps',
        nargs=2,
        type=str,
        metavar=('png_vx', 'png_vy'),
        default=['', ''],
        help='Water maps',
    )
    parser.add_argument(
        '--ground_height',
        type=float,
        default=None,
        help='Ground height',
    )
    parser.add_argument(
        '--control_type',
        type=str,
        default='position',
        help='Control type',
    )
    parser.add_argument(
        '--torque_equation',
        type=str,
        default=None,
        help='Torque equation',
    )
    parser.add_argument(
        '--save_to_models',
        action='store_true',
        help='Save data to farms_models_data',
    )
    parser.add_argument(
        '--drive_config',
        type=str,
        default=None,
        help='Descending drive config',
    )
    parser.add_argument(
        '--max_torque',
        type=float,
        default=np.inf,
        help='Max torque',
    )
    parser.add_argument(
        '--max_velocity',
        type=float,
        default=np.inf,
        help='Max velocity',
    )
    parser.add_argument(
        '--lateral_friction',
        type=float,
        default=1.0,
        help='Lateral friction',
    )
    parser.add_argument(
        '--feet_friction',
        nargs='+',
        type=float,
        default=None,
        help='Feet friction',
    )
    parser.add_argument(
        '--default_restitution',
        type=float,
        default=0.0,
        help='Default restitution',
    )
    parser.add_argument(
        '--viscosity',
        type=float,
        default=1.0,
        help='Viscosity',
    )
    parser.add_argument(
        '--self_collisions',
        action='store_true',
        help='Apply self collisions',
    )
    parser.add_argument(
        '--spawn_loader',
        type=str,
        choices=('FARMS', 'PYBULLET'),
        default='FARMS',
        help='Spawn loader',
    )
    parser.add_argument(
        '--simulator',
        type=str,
        choices=('MUJOCO', 'PYBULLET'),
        default='MUJOCO',
        help='Simulator',
    )
    for key in options_kwargs_float_keys():
        parser.add_argument(
            f'--{key}',
            type=float,
            default=None,
            help=f'{key}',
        )
    for key in options_kwargs_float_list_keys():
        parser.add_argument(
            f'--{key}',
            nargs='+',
            type=float,
            default=None,
            help=f'{key}',
        )
    for key in options_kwargs_int_keys():
        parser.add_argument(
            f'--{key}',
            type=int,
            default=None,
            help=f'{key}',
        )
    for key in options_kwargs_int_list_keys():
        parser.add_argument(
            f'--{key}',
            nargs='+',
            type=int,
            default=None,
            help=f'{key}',
        )
    for key in options_kwargs_str_keys():
        parser.add_argument(
            f'--{key}',
            type=str,
            default=None,
            help=f'{key}',
        )
    for key in options_kwargs_str_list_keys():
        parser.add_argument(
            f'--{key}',
            nargs='+',
            type=str,
            default=None,
            help=f'{key}',
        )
    for key in options_kwargs_bool_keys():
        parser.add_argument(
            f'--{key}',
            type=bool,
            default=None,
            help=f'{key}',
        )
    for key in options_kwargs_sph_float_keys():
        parser.add_argument(
            f'--{key}',
            type=float,
            default=None,
            help=f'{key}',
        )
    return parser


def config_parse_args():
    """Parse args"""
    parser = config_argument_parser()
    return parser.parse_args()


def parser_model_gen(description='Generate model'):
    """Parse args"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--animat',
        type=str,
        default='',
        help='Animat name',
    )
    parser.add_argument(
        '--version',
        type=str,
        default='',
        help='Animat version',
    )
    parser.add_argument(
        '--sdf_path',
        type=str,
        default='',
        help='SDF path',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='',
        help='Model directory path',
    )
    parser.add_argument(
        '--original',
        type=str,
        default='',
        help='Original file',
    )
    parser.add_argument(
        '--output_gen_config',
        type=str,
        help='Generation config output',
    )
    return parser


def parse_args_model_gen(*args, **kwargs):
    """Parse args"""
    parser = parser_model_gen(*args, **kwargs)
    return parser.parse_args()


def parser_postprocessing(description='Amphibious simulation post-processing'):
    """Parse args"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=(
            lambda prog:
            argparse.HelpFormatter(prog, max_help_position=50)
        ),
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Animat data',
    )
    parser.add_argument(
        '--animat',
        type=str,
        help='Animat options',
    )
    parser.add_argument(
        '--simulation',
        type=str,
        help='Simulation options',
    )
    parser.add_argument(
        '--drive_config',
        type=str,
        default='',
        help='Descending drive method',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path',
    )
    return parser


def parse_args_postprocessing(*args, **kwargs):
    """Parse args"""
    parser = parser_postprocessing(*args, **kwargs)
    return parser.parse_args()


def parser_sweep(description='Plot amphibious sweep') -> argparse.ArgumentParser:
    """Parse args"""
    parser = argparse.ArgumentParser(
        description=description,
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
        '--names',
        metavar='name1 name2 ...',
        type=str,
        nargs='+',
        default=[],
        help='Sweep experiments names',
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
    return parser


def validate_sweep_clargs(clargs):
    """Validate sweep command line arguments"""
    assert len(clargs.logs) == len(clargs.names), (
        f'{len(clargs.logs)=} != {len(clargs.names)=}'
    )
    assert len(clargs.logs) == len(clargs.labels), (
        f'{len(clargs.logs)=} != {len(clargs.labels)=}'
    )


def parse_args_sweep(*args, **kwargs):
    """Parse args"""
    parser = parser_sweep(*args, **kwargs)
    clargs = parser.parse_args()
    validate_sweep_clargs(clargs)
    return clargs
