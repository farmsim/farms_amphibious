"""Callbacks"""

import numpy as np

from farms_mujoco.swimming.callback import SwimmingCallback


def setup_callbacks(
        animats_data,
        animats_options,
        arena_options,
        camera=None,
        water_properties=None,
):
    """Callbacks for amphibious simulation"""
    callbacks = []
    if arena_options.water.sdf:
        callbacks += [
            SwimmingCallback(
                animat_i,
                animat_data,
                animat_options,
                arena_options,
                water_properties=water_properties,
            )
            for animat_i, (animat_data, animat_options) in enumerate(zip(
                    animats_data,
                    animats_options,
            ))
        ]
    if camera is not None:
        callbacks += [camera]
    return callbacks


def water_velocity_from_maps(position, water_maps):
    """Water velocity from maps"""
    vel = np.zeros(3)
    if all(
            water_maps['pos_min'][i] < position[i] < water_maps['pos_max'][i]
            for i in range(2)
    ):
        vel[:2] = [
            water_maps[png][tuple(
                (
                    max(0, min(
                        water_maps[png].shape[index]-1,
                        round(water_maps[png].shape[index]*(
                            (
                                position[index]
                                - water_maps['pos_min'][index]
                            ) / (
                                water_maps['pos_max'][index]
                                - water_maps['pos_min'][index]
                            )
                        ))
                    ))
                )
                for index in range(2)
            )]
            for png_i, png in enumerate(['vel_x', 'vel_y'])
        ]
    # vel[1] *= -1
    return vel


def maps_surface_callback(surface_height):
    """Maps surface callback"""
    def surface_callback(t, x, y):  # pylint: disable=unused-argument
        """Surface"""
        return surface_height
    return surface_callback


def maps_density_callback(density):
    """Maps density callback"""
    def density_callback(t, x, y, z):  # pylint: disable=unused-argument
        """Density"""
        return density
    return density_callback


def maps_viscosity_callback(viscosity):
    """Maps viscosity callback"""
    def viscosity_callback(t, x, y, z):  # pylint: disable=unused-argument
        """Viscosity"""
        return viscosity
    return viscosity_callback


def maps_velocity_callback(water_maps):
    """Maps velocity callback"""
    def velocity_callback(t, x, y, z):  # pylint: disable=unused-argument
        """Velocity in global frame"""
        return water_velocity_from_maps(
            position=[x, y, z],
            water_maps=water_maps,
        )
    return velocity_callback
