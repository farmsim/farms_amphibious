"""Amphibious"""

import pybullet
from farms_core.model.control import AnimatController
from farms_bullet.model.animat import Animat


class Amphibious(Animat):
    """Amphibious animat"""

    def __init__(
            self,
            controller: AnimatController,
            timestep: float,
            iterations: int,
            **kwargs,
    ):
        super().__init__(
            data=controller.animat_data if controller is not None else None,
            controller=controller,
            **kwargs,
        )
        self.timestep: float = timestep
        self.n_iterations: int = iterations
        self.xfrc_plot: bool = None

    def spawn(self):
        """Spawn amphibious"""
        super().spawn()

        # Links masses
        link_mass_multiplier = {
            link.name: link.mass_multiplier
            for link in self.options.morphology.links
        }
        for link, index in self.links_map.items():
            if link in link_mass_multiplier:
                mass, _, torque, *_ = pybullet.getDynamicsInfo(
                    bodyUniqueId=self.identity(),
                    linkIndex=index,
                )
                pybullet.changeDynamics(
                    bodyUniqueId=self.identity(),
                    linkIndex=index,
                    mass=link_mass_multiplier[link]*mass,
                )
                pybullet.changeDynamics(
                    bodyUniqueId=self.identity(),
                    linkIndex=index,
                    localInertiaDiagonal=link_mass_multiplier[link]*torque,
                )

        # Debug
        self.xfrc_plot = [
            [
                False,
                pybullet.addUserDebugLine(
                    lineFromXYZ=[0, 0, 0],
                    lineToXYZ=[0, 0, 0],
                    lineColorRGB=[0, 0, 0],
                    lineWidth=3*self.units.meters,
                    lifeTime=0,
                    parentObjectUniqueId=self.identity(),
                    parentLinkIndex=i
                )
            ]
            for i in range(self.data.sensors.xfrc.array.shape[1])
        ] if self.options.show_xfrc else []
