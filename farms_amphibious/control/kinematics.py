"""Kinematics"""

import numpy as np
from scipy.interpolate import interp1d
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.model.control import AnimatController, ControlType
from farms_core.experiment.options import ExperimentOptions


def kinematics_interpolation(
        kin_times,
        kinematics,
        timestep,
        n_iterations,
):
    """Kinematics interpolations"""
    simulation_duration = timestep*n_iterations
    sim_times = np.arange(0, simulation_duration, timestep)
    assert len(kin_times) == kinematics.shape[0], (
        f'{len(kin_times)=} != {kinematics.shape[0]=}'
    )
    return interp1d(
        kin_times,
        kinematics,
        axis=0
    )(sim_times)


class KinematicsController(AnimatController):
    """Amphibious kinematics"""

    def __init__(
            self,
            joints_names,
            kinematics,
            sampling,
            timestep,
            n_iterations,
            animat_data,
            max_torques,
            invert_motors=False,
            indices=None,
            time_index=None,
            degrees=False,
            init_time=0,
            end_time=0,
    ):
        super().__init__(
            joints_names=joints_names,
            max_torques=max_torques,
            muscles_names=[],
        )

        # Time vector
        if time_index is not None:
            time_vector = kinematics[:, time_index]
            time_vector -= time_vector[0]
        else:
            data_duration = kinematics.shape[0]*sampling
            time_vector = np.arange(0, data_duration, sampling)

        # Indices
        if indices:
            kinematics = kinematics[:, indices]
        elif time_index:
            mask = np.ones(kinematics.shape, dtype=bool)
            mask[:, time_index] = False
            kinematics = kinematics[mask]
        assert kinematics.shape[1] == len(joints_names[ControlType.POSITION]), (
            f'Expected {len(joints_names[ControlType.POSITION])} joints,'
            f' but got {kinematics.shape[1]} (shape={kinematics.shape}'
            f', indices={indices})'
        )

        # Converting to radians
        if degrees:
            kinematics = np.deg2rad(kinematics)

        # Invert motors
        if invert_motors:
            kinematics *= -1

        # Add initial time
        if init_time > 0:
            kinematics = np.insert(
                arr=kinematics,
                obj=0,
                values=np.repeat(
                    a=[kinematics[0, :]],
                    repeats=int(init_time/sampling)+1,
                    axis=0,
                ),
                axis=0,
            )
            time_vector += init_time
            time_vector = np.insert(
                time_vector,
                obj=0,
                values=np.linspace(
                    0,
                    time_vector[0],
                    int(init_time/sampling)+1,
                ),
            )

        # Add end time
        if end_time > 0:
            kinematics = np.insert(
                arr=kinematics,
                obj=kinematics.shape[0],
                values=np.repeat(
                    a=[kinematics[-1, :]],
                    repeats=int(end_time/sampling)+1,
                    axis=0,
                ),
                axis=0,
            )
            if time_vector is not None:
                time_vector = np.insert(
                    arr=time_vector,
                    obj=time_vector.shape[0],
                    values=np.linspace(
                        time_vector[-1]+timestep,
                        time_vector[-1]+end_time,
                        int(end_time/sampling)+1,
                    ),
                )

        self.kinematics = kinematics_interpolation(
            kin_times=time_vector,
            kinematics=kinematics,
            timestep=timestep,
            n_iterations=n_iterations,
        )
        self.animat_data = animat_data

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):
        """From options"""
        joints_names = animat_options.control.joints_names()
        joints_control_types = {
            motor.joint_name: ControlType.from_string_list(
                motor.control_types,
            )
            for motor in animat_options.control.motors
        }
        joints_names_per_type = AnimatController.joints_from_control_types(
            joints_names=joints_names,
            joints_control_types=joints_control_types,
        )
        max_torques = {
            motor.joint_name: motor.limits_torque[1]
            for motor in animat_options.control.motors
        }
        max_torques_per_type = AnimatController.max_torques_from_control_types(
            joints_names=joints_names,
            max_torques=max_torques,
            joints_control_types=joints_control_types,
        )
        return KinematicsController(
            joints_names=joints_names_per_type,
            kinematics=np.genfromtxt(config['kinematics_file'], delimiter=','),
            sampling=config['kinematics_sampling'],
            indices=config['kinematics_indices'],
            time_index=config['kinematics_time_index'],
            invert_motors=config['kinematics_invert'],
            degrees=config['kinematics_degrees'],
            timestep=experiment_options.simulation.physics.timestep,
            n_iterations=experiment_options.simulation.runtime.n_iterations,
            animat_data=animat_data,
            max_torques=max_torques_per_type,
            init_time=config['kinematics_start'],
            end_time=config['kinematics_end'],
        )

    def positions(self, iteration, time, timestep):
        """Postions"""
        return dict(zip(
            self.joints_names[ControlType.POSITION],
            self.kinematics[iteration],
        ))
