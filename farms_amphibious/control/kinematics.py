"""Kinematics"""

import numpy as np
from scipy.interpolate import interp1d
from farms_core.model.control import AnimatController, ControlType


def kinematics_interpolation(
        kinematics,
        sampling,
        timestep,
        n_iterations,
        time_vector=None,
):
    """Kinematics interpolations"""
    data_duration = (
        time_vector[-1]
        if time_vector is not None
        else sampling*kinematics.shape[0]
    )
    simulation_duration = timestep*n_iterations
    interp_x = (
        time_vector
        if time_vector is not None
        else np.arange(0, data_duration, sampling)
    )
    interp_xn = np.arange(0, simulation_duration, timestep)
    assert data_duration >= simulation_duration, (
        f'Data {data_duration} < {simulation_duration} Sim'
    )
    assert len(interp_x) == kinematics.shape[0], (
        f'{len(interp_x)} != {kinematics.shape[0]} (shape={kinematics.shape})'
    )
    assert interp_x[-1] >= interp_xn[-1], (
        f'Data[-1] {interp_x[-1]} < {interp_xn[-1]} Sim[-1]'
    )
    return interp1d(
        interp_x,
        kinematics,
        axis=0
    )(interp_xn)


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
        # Time index
        if time_index is not None:
            time_vector = kinematics[:, time_index]
            time_vector -= time_vector[0]
        else:
            time_vector = None
        # Indices
        if indices:
            kinematics = kinematics[:, indices]
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
            kinematics=kinematics,
            sampling=sampling,
            timestep=timestep,
            n_iterations=n_iterations,
            time_vector=time_vector,
        )
        self.animat_data = animat_data

    def positions(self, iteration, time, timestep):
        """Postions"""
        return dict(zip(
            self.joints_names[ControlType.POSITION],
            self.kinematics[iteration],
        ))
