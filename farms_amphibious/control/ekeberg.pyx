"""Ekeberg muscle model"""

include 'sensor_convention.pxd'
cimport numpy as np
import numpy as np


cdef enum:

    ALPHA = 0
    BETA = 1
    GAMMA = 2
    DELTA = 3
    EPSILON = 4


cdef inline double sign(double value):
    """Sign"""
    if value < 0:
        return -1
    else:
        return 1


cdef class EkebergMuscleCy(JointsMusclesCy):
    """Ekeberg muscle model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activations = np.zeros(self.n_joints, dtype=np.double)
        self.joints_offsets = np.zeros(self.n_joints, dtype=np.double)
        self.spring_coefs = np.zeros(self.n_joints, dtype=np.double)
        self.damping_coefs = np.zeros(self.n_joints, dtype=np.double)

    cpdef void update_activations(self, unsigned int iteration):
        """Update offsets"""
        self.activations = self.state.outputs(iteration)

    cpdef void update_offsets(self, unsigned int iteration):
        """Update offsets"""
        self.joints_offsets = np.array(self.state.offsets(iteration))

    cpdef void step(self, unsigned int iteration):
        """Step"""
        cdef unsigned int muscle_i, joint_data_i, osc_0, osc_1
        cdef DTYPE neural_diff, neural_sum
        cdef DTYPE active_torque, stiffness_intermediate
        cdef DTYPE active_stiffness, passive_stiffness, damping, friction
        cdef DTYPEv1 positions = self.joints_data.positions(iteration)
        cdef DTYPEv1 velocities = self.joints_data.velocities(iteration)

        # Update
        self.update_activations(iteration)
        self.update_offsets(iteration)

        # For each muscle
        for muscle_i in range(self.n_joints):

            # Muscle indices map - The indices relate to the list of motors
            # defined in the animat config, with only the motors using the
            # Ekeberg muscle model being present. The given indices, in the
            # order of muscles, will provide the mapping to the indices of the
            # joints sensors data. See the amphibious controller initialisation
            # AmphibiousController.__init__ for the declaration.
            joint_data_i = self.indices[muscle_i]

            # Oscillator indices - The indices relate to the list
            # animat_data.network.oscillators.names defined in the animat
            # config. The first index accesses the first or second input, while
            # the second index accesses the index for the neural activity. Refer
            # to MusclesMap.__init__ for the declaration.
            osc_0 = self.osc_indices[0][muscle_i]
            osc_1 = self.osc_indices[1][muscle_i]

            # Data
            neural_diff = self.activations[osc_1] - self.activations[osc_0]
            neural_sum = self.activations[osc_0] + self.activations[osc_1]
            m_delta_phi = self.joints_offsets[muscle_i] - (
                positions[joint_data_i] - self.transform_bias[joint_data_i]
            )/self.transform_gain[joint_data_i]  # Amphibious convention space

            # Torques
            active_torque = (
                self.parameters[muscle_i][ALPHA]
                *neural_diff
                *self.transform_gain[joint_data_i]  # SDF space
            )
            stiffness_intermediate = (
                self.parameters[muscle_i][BETA]
                *m_delta_phi
            )
            active_stiffness = (
                neural_sum
                *stiffness_intermediate
                *self.transform_gain[joint_data_i]  # SDF space
            )
            passive_stiffness = (
                self.parameters[muscle_i][GAMMA]
                *stiffness_intermediate
                *self.transform_gain[joint_data_i]  # SDF space
            )
            damping = -(
                self.parameters[muscle_i][DELTA]
                *velocities[joint_data_i]
            )
            friction = -(
                self.parameters[muscle_i][EPSILON]
                *sign(velocities[joint_data_i])
            )

            # Coefficients
            self.damping_coefs[muscle_i] = self.parameters[muscle_i][DELTA]
            self.spring_coefs[muscle_i] = self.parameters[muscle_i][BETA]*(
                neural_sum + self.parameters[muscle_i][GAMMA]
            )

            # Transform to SDF space
            self.joints_offsets[muscle_i] = (
                self.transform_gain[joint_data_i]
                *self.joints_offsets[muscle_i]
                + self.transform_bias[joint_data_i]
            )

            # Log
            torque = active_torque + active_stiffness + passive_stiffness + damping + friction
            self.joints_data.array[iteration, joint_data_i, JOINT_CMD_TORQUE] = torque
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_ACTIVE] = active_torque  #  + active_stiffness
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_STIFFNESS] = passive_stiffness
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_DAMPING] = damping
            self.joints_data.array[iteration, joint_data_i, JOINT_TORQUE_FRICTION] = friction
