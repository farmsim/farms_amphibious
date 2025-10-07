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
