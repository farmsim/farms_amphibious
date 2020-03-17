"""GPS for amphibious animat"""

from farms_bullet.sensors.sensors import LinksStatesSensor


class AmphibiousGPS(LinksStatesSensor):
    """Amphibious GPS"""

    def __init__(self, array, animat_id, links, options, units):
        super(AmphibiousGPS, self).__init__(
            array=array,
            animat_id=animat_id,
            links=links,
            units=units
        )
        self.options = options

    def update(self, iteration):
        """Update sensor"""
        if self.options.collect_gps:
            self.collect(iteration, self.links)
        # elif self.options.control.drives.forward > 3:
        #     self.collect(iteration, self.links[:12])
        # else:
        #     self.collect(iteration, [self.links[0]])
        else:
            self.collect(iteration, self.links[:self.options.morphology.n_joints_body])
