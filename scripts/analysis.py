"""Analyse events"""

import os

import numpy as np
from scipy.signal import find_peaks

from farms_core.io.sdf import ModelSDF
from farms_core.io.yaml import pyobject2yaml
from farms_core.utils.profile import profile
from farms_core.simulation.options import SimulationOptions
from farms_core.analysis.metrics import (
    com_velocities,
    average_2d_velocity,
    compute_torque_integral,
)

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.model.convention import AmphibiousConvention
from farms_amphibious.analysis.metrics import analyse_gait_amphibious
from farms_amphibious.utils.parse_args import parser_postprocessing


def main():
    """Main"""

    # Clargs
    parser = parser_postprocessing(description='Analyse amphibious experiment')
    parser.add_argument(
        '--bodylength',
        type=float,
        default=None,
        help='Body length',
    )
    clargs = parser.parse_args()

    # Load data
    animat_options = AmphibiousOptions.load(clargs.animat)
    simulation_options = SimulationOptions.load(clargs.simulation)
    animat_data = AmphibiousData.from_file(clargs.data)
    n_iterations = simulation_options.n_iterations
    iteration_0 = n_iterations//2
    iteration_1 = n_iterations-2
    timestep = animat_data.timestep

    # Plot simulation data
    times = simulation_options.times()
    assert len(times) == n_iterations, f'{len(times)=} != {n_iterations=}'
    times = times[:animat_data.sensors.links.array.shape[0]]
    analysis = {}

    # Convention
    osc_names = animat_data.network.oscillators.names
    convention = AmphibiousConvention.from_amphibious_options(animat_options)
    n_oscillators = len(osc_names)
    all_indices = list(range(n_oscillators))
    n_osc_body = convention.n_osc_body()
    n_osc_legs = convention.n_osc_legs()
    body_indices = list(range(n_osc_body))
    body_left_indices =  [i for i in body_indices if not i%2]
    body_right_indices =  [i for i in body_indices if i%2]
    assert n_osc_body+n_osc_legs == n_oscillators, (
        f'{n_osc_body=}, {n_osc_legs=}, {n_oscillators=}'
    )
    limb_indices = list(range(n_osc_body, n_osc_body+n_osc_legs))

    # Body
    analysis['morphology'] = {}
    analysis['morphology']['bodylength'] = clargs.bodylength
    model = ModelSDF.read(filename=animat_options.sdf)[0]
    mass = analysis['morphology']['mass'] = model.mass()
    gravity = float(np.linalg.norm(simulation_options.gravity))

    # Metrics
    metrics = analysis['metrics'] = {}

    # Frequency
    phases = np.array(animat_data.state.phases_all())
    phases_cut = phases[iteration_0:iteration_1, :]
    oscillators_frq = np.diff(phases_cut, axis=0)/(2*np.pi*timestep)
    metrics['frequency'] = {
        'all': float(np.mean(oscillators_frq[:, all_indices])),
        'body': float(np.mean(oscillators_frq[:, body_indices])),
        'body_left': float(np.mean(oscillators_frq[:, body_left_indices])),
        'body_right': float(np.mean(oscillators_frq[:, body_right_indices])),
        'limb': float(np.mean(oscillators_frq[:, limb_indices])),
    }

    # Velocity
    data_links = animat_data.sensors.links
    positions0 = np.array(data_links.urdf_positions()[:, 0, :])
    metrics['velocity0'] = float(average_2d_velocity(
        positions=positions0,
        iterations=[iteration_0, iteration_1],
        timestep=timestep,
    ))
    average_velocity = metrics['velocity'] = float(np.mean(np.linalg.norm(
        com_velocities(
            data_links=data_links,
            iterations=[iteration_0, iteration_1],
            timestep=timestep,
        ),
        axis=-1,
    )))

    # Torques integral
    data_joints = animat_data.sensors.joints
    for exponent in range(1, 5):
        metrics[f'torque_integral{exponent}'] = float(compute_torque_integral(
            data_joints=data_joints,
            iteration0=iteration_0,
            iteration1=iteration_1,
            exponent=exponent,
            times=times,
            timestep=timestep
        ))

    # CoT
    average_mechanical_work = float(np.sum(
        data_joints.mechanical_power()
    )*timestep/(times[iteration_1] - times[iteration_0]))
    metrics['cot'] = average_mechanical_work/(mass*gravity*average_velocity)

    # Gaits
    metrics['gait'] = {}
    gait_analysis = analyse_gait_amphibious(animat_data, animat_options)
    for key, value in gait_analysis.items():
        metrics['gait'][key] = float(value)

    # Get Events
    events = analysis['events'] = {}
    events['joint_max_pos'] = {}
    joints_pos = np.array(animat_data.sensors.joints.positions_all())
    max_frequency = 10
    distance = round(1/(max_frequency*timestep))
    joint_indices = [convention.bodyjoint2index(joint_i=0)]
    if convention.n_legs:
        joint_indices.append(
            convention.legjoint2index(leg_i=0, side_i=0, joint_i=0)
        )
    for joint_index in joint_indices:
        events['joint_max_pos'][joint_index], _props = find_peaks(
            x=joints_pos[:, joint_index],
            height=(0, None),  # [rad]
            distance=distance,  # [iterations]
            prominence=0.1,  # [rad]
        )
        events['joint_max_pos'][joint_index] = (
            events['joint_max_pos'][joint_index].tolist()
        )

    # Save events times
    pyobject2yaml(
        filename=os.path.join(clargs.output, 'analysis.yaml'),
        pyobject=analysis,
    )


if __name__ == '__main__':
    profile(main)
