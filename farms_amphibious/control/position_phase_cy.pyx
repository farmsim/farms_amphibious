"""Position phase model"""

include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np
from libc.math cimport sin, M_PI, fmod


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
        self.threshold = kwargs.pop('threshold', 0)
        super().__init__(**kwargs)

    cpdef void step(self, unsigned int iteration):
        """Step"""
        cdef double pos, pos_raw, dif
        cdef unsigned int joint_i, joint_data_i, osc_i_0, osc_i_1
        cdef DTYPEv1 offsets = self.state.offsets(iteration)
        cdef DTYPEv1 phases = self.state.phases(iteration)
        cdef DTYPEv1 amplitudes = self.state.amplitudes(iteration)

        # For each joint
        for joint_i in range(self.n_joints):

            # Data
            joint_data_i = self.indices[joint_i]
            pos_raw = self.joints_data.array[iteration, joint_data_i, JOINT_POSITION]
            pos = (  # Amphibious convention space
                pos_raw - self.transform_bias[joint_data_i]
            )/self.transform_gain[joint_data_i]
            osc_i_0 = self.osc_indices[0][joint_i]
            osc_i_1 = self.osc_indices[1][joint_i]
            assert osc_i_0 < len(phases), (
                f'{osc_i_0=} !< {len(phases)=}'
                f'\n{joint_data_i=}'
                f'\n{np.array(phases)=}'
                f'\n{np.array(self.osc_indices)=}'
            )
            assert osc_i_1 >= len(phases), (
                f'{osc_i_1=} !>= {len(phases)=}'
                f'\n{joint_data_i=}'
                f'\n{np.array(phases)=}'
                f'\n{np.array(self.osc_indices)=}'
            )

            if amplitudes[osc_i_0] < self.threshold:  # Swimming
                desired_angle =  0 + offsets[joint_data_i]
            else:  # Walking
                desired_angle =  phases[osc_i_0] + offsets[joint_data_i]
            dif = fmod(desired_angle - pos + M_PI, 2*M_PI) - M_PI
            if dif < -M_PI:
                dif += 2*M_PI
            self.joints_data.array[iteration, joint_data_i, JOINT_CMD_POSITION] = (
                self.transform_gain[joint_data_i]*(
                    dif + pos
                ) + self.transform_bias[joint_data_i]
            )
