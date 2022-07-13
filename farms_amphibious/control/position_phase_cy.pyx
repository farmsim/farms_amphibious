"""Position phase model"""

include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np
from libc.math cimport sin, M_PI


cdef class PositionPhaseCy(JointsControlCy):
    """Position phase model"""

    def __init__(
            self,
            OscillatorNetworkStateCy state,
            UITYPEv2 osc_indices,
            **kwargs,
    ):
        self.state = state
        self.osc_indices = osc_indices
        self.weight = kwargs.pop('weight', 0)
        self.offset = kwargs.pop('offset', 0)
        self.threshold = kwargs.pop('threshold', 0)
        super().__init__(**kwargs)

    cpdef void step(self, unsigned int iteration):
        """Step"""
        cdef double pos, dif
        cdef unsigned int joint_i, joint_data_i, osc_i
        cdef DTYPEv1 offsets = self.state.offsets(iteration)
        cdef DTYPEv1 phases = self.state.phases(iteration)
        cdef DTYPEv1 amplitudes = self.state.amplitudes(iteration)

        # For each joint
        for joint_i in range(self.n_joints):

            # Data
            joint_data_i = self.indices[joint_i]
            pos = self.joints_data.array[iteration, joint_data_i, JOINT_POSITION]
            osc_i = self.osc_indices[0][joint_data_i]
            assert osc_i < len(phases)

            if amplitudes[osc_i] < self.threshold:  # Swimming
                dif = (M_PI - pos) % (2*M_PI) - M_PI
                if dif < - M_PI:
                    dif += 2*M_PI
            else:  # Walking
                dif = (phases[osc_i] - pos + M_PI) % (2*M_PI) - M_PI
                if dif < - M_PI:
                    dif += 2*M_PI
            self.joints_data.array[iteration, joint_data_i, JOINT_CMD_POSITION] = (
                self.transform_gain[joint_data_i]*(
                    dif + pos + offsets[joint_data_i]
                ) + self.transform_bias[joint_data_i]
            )
