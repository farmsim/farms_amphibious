"""Kinematics"""

import numpy as np
from scipy.interpolate import interp1d
from farms_core.model.control import AnimatController, ControlType


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

    def positions(self, iteration, time, timestep):
        """Postions"""
        return dict(zip(
            self.joints_names[ControlType.POSITION],
            self.kinematics[iteration],
        ))
