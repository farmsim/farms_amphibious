"""Callbacks"""

from farms_core.sensors.sensor_convention import sc
from farms_mujoco.swimming.drag import SwimmingHandler
from farms_mujoco.simulation.task import TaskCallback
from .model.options import AmphibiousOptions


def setup_callbacks(animat_options, camera=None):
    """Callbacks for amphibious simulation"""
    callbacks = []
    if animat_options.physics.drag or animat_options.physics.sph:
        callbacks += [
            SwimmingCallback(animat_options),
        ]
    if camera is not None:
        callbacks += [camera]
    return callbacks


class SwimmingCallback(TaskCallback):
    """Swimming callback"""

    def __init__(self, animat_options: AmphibiousOptions):
        super().__init__()
        self.animat_options = animat_options
        self._handler: SwimmingHandler = None

    def initialize_episode(self, task, physics):
        """Initialize episode"""
        self._handler = SwimmingHandler(
            data=task.data,
            animat_options=self.animat_options,
            units=task.units,
            physics=physics,
        )

    def before_step(self, task, action, physics):
        """Step hydrodynamics"""
        self._handler.step(task.iteration)
        # physics.data.xfrc_applied[:, :] = 0  # Reset all forces
        indices = task.maps['sensors']['data2xfrc']
        physics.data.xfrc_applied[indices, :] = (
            task.data.sensors.xfrc.array[
                task.iteration, :,
                sc.xfrc_force_x:sc.xfrc_torque_z+1,
            ]
        )
        for force_i, (rotation_mat, force_local) in enumerate(zip(
                physics.data.xmat[indices],
                physics.data.xfrc_applied[indices],
        )):
            physics.data.xfrc_applied[indices[force_i]] = (
                rotation_mat.reshape([3, 3])  # Local to global frame
                @ force_local.reshape([3, 2], order='F')
            ).flatten(order='F')
        physics.data.xfrc_applied[indices, :3] *= task.units.newtons
        physics.data.xfrc_applied[indices, 3:] *= task.units.torques
