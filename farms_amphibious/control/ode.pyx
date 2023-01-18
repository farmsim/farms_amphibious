"""Cython code"""

# import time
# import numpy as np
# cimport numpy as np

# cimport cython
# from cython.parallel import prange

from libc.math cimport sin, cos, fabs
# from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

from ..data.data_cy cimport ConnectionType


cdef inline DTYPE phase(
    DTYPEv1 state,
    unsigned int index
) nogil:
    """Phase"""
    return state[index]


cdef inline DTYPE amplitude(
    DTYPEv1 state,
    unsigned int index,
    unsigned int n_oscillators,
) nogil:
    """Amplitude"""
    return state[index+n_oscillators]


cdef inline DTYPE joint_offset(
    DTYPEv1 state,
    unsigned int index,
    unsigned int n_oscillators,
) nogil:
    """Joint offset"""
    return state[index+2*n_oscillators]




cpdef inline void ode_dphase(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
    OscillatorsConnectivityCy connectivity,
) nogil:
    """Oscillator phase ODE

    d_theta = (
        omega*(1 + omega_mod_amp*cos(theta + omega_mod_phase))
        + sum amplitude_j*weight*sin(phase_j - phase_i - phase_bias)
    )

    """
    cdef unsigned int i, i0, i1, n_oscillators = oscillators.n_oscillators
    for i in range(n_oscillators):
        # Intrinsic frequency
        dstate[i] = oscillators.c_angular_frequency(iteration, i, drives)
        if oscillators.c_modular_amplitudes(i) > 1e-3:
            dstate[i] *= (
                1 + oscillators.c_modular_amplitudes(i)*cos(
                    phase(state, i) + oscillators.c_modular_phases(i)
                )
            )
    for i in range(connectivity.n_connections):
        # Neural couplings
        i0 = connectivity.connections.array[i, 0]
        i1 = connectivity.connections.array[i, 1]
        dstate[i0] += state[n_oscillators+i1]*connectivity.c_weight(i)*sin(
            phase(state, i1) - phase(state, i0)
            - connectivity.c_desired_phase(i)
        )


cpdef inline void ode_damplitude(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
) nogil:
    """Oscillator amplitude ODE

    d_amplitude = rate*(nominal_amplitude - amplitude)

    """
    cdef unsigned int i, n_oscillators = oscillators.n_oscillators
    for i in range(n_oscillators):  # , nogil=True):
        # rate*(nominal_amplitude - amplitude)
        dstate[n_oscillators+i] = oscillators.c_rate(i)*(
            oscillators.c_nominal_amplitude(iteration, i, drives)
            - amplitude(state, i, n_oscillators)
        )


cpdef inline void ode_stretch(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    JointSensorArrayCy joints,
    JointsConnectivityCy joints2osc_map,
    unsigned int n_oscillators,
) nogil:
    """Sensory feedback - Stretch

    Can affect d_phase

    """
    cdef unsigned int i, i0, i1, connection_type
    for i in range(joints2osc_map.n_connections):
        i0 = joints2osc_map.connections.array[i, 0]  # Oscillator
        i1 = joints2osc_map.connections.array[i, 1]  # Joint
        connection_type = joints2osc_map.connections.array[i, 2]
        if connection_type == ConnectionType.STRETCH2FREQTEGOTAE:
            # stretch_weight*joint_position*sin(phase)
            dstate[i0] += (
                joints2osc_map.c_weight(i)
                *joints.position_cy(iteration, i1)
                *sin(state[i0])  # For Tegotae
            )
        elif connection_type == ConnectionType.STRETCH2AMPTEGOTAE:
            # stretch_weight*joint_position*sin(phase)
            dstate[n_oscillators+i0] += (
                joints2osc_map.c_weight(i)
                *joints.position_cy(iteration, i1)
                *sin(state[i0])  # For Tegotae
            )
        elif connection_type == ConnectionType.STRETCH2FREQ:
            # stretch_weight*joint_position  # *sin(phase)
            dstate[i0] += (
                joints2osc_map.c_weight(i)
                *joints.position_cy(iteration, i1)
            )
        elif connection_type == ConnectionType.STRETCH2AMP:
            # stretch_weight*joint_position  # *sin(phase)
            dstate[n_oscillators+i0] += (
                joints2osc_map.c_weight(i)
                *joints.position_cy(iteration, i1)
            )
        else:
            printf(
                'Joint connection %i of type %i is incorrect'
                ', should be %i, %i, %i or %i\n',
                i,
                connection_type,
                ConnectionType.STRETCH2FREQ,
                ConnectionType.STRETCH2AMP,
                ConnectionType.STRETCH2FREQTEGOTAE,
                ConnectionType.STRETCH2AMPTEGOTAE,
            )


cpdef inline void ode_contacts(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactsConnectivityCy contacts2osc_map,
) nogil:
    """Sensory feedback - Contacts

    Can affect d_phase and d_amplitude

    """
    cdef DTYPE contact_reaction
    cdef unsigned int i, i0, i1, connection_type
    for i in range(contacts2osc_map.n_connections):
        i0 = contacts2osc_map.connections.array[i, 0]
        i1 = contacts2osc_map.connections.array[i, 1]
        connection_type = contacts2osc_map.connections.array[i, 2]
        contact_reaction = fabs(contacts.c_reaction_z(iteration, i1))
        if connection_type == ConnectionType.REACTION2FREQ:
            dstate[i0] += (
                contacts2osc_map.c_weight(i)
                *contact_reaction
            )
        elif connection_type == ConnectionType.REACTION2FREQTEGOTAE:
            dstate[i0] += (
                contacts2osc_map.c_weight(i)
                *contact_reaction
                # *cos(state[i0])
                *sin(state[i0])  # For Tegotae
            )


cpdef inline void ode_xfrc(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    XfrcArrayCy xfrc,
    XfrcConnectivityCy xfrc2osc_map,
    unsigned int n_oscillators,
) nogil:
    """Sensory feedback - Xfrc

    Can affect d_phase and d_amplitude

    """
    cdef DTYPE xfrc_force
    cdef unsigned int i, i0, i1, connection_type
    for i in range(xfrc2osc_map.n_connections):
        i0 = xfrc2osc_map.connections.array[i, 0]
        i1 = xfrc2osc_map.connections.array[i, 1]
        connection_type = xfrc2osc_map.connections.array[i, 2]
        xfrc_force = fabs(xfrc.c_force_y(iteration, i1))
        if connection_type == ConnectionType.LATERAL2FREQ:
            # dfrequency += xfrc_weight*xfrc_force
            dstate[i0] += (
                xfrc2osc_map.c_weights(i)*xfrc_force
            )
        elif connection_type == ConnectionType.LATERAL2AMP:
            # damplitude += xfrc_weight*xfrc_force
            dstate[n_oscillators+i0] += (
                xfrc2osc_map.c_weights(i)*xfrc_force
            )
        else:
            printf(
                'Xfrc connection %i of type %i is incorrect'
                ', should be %i or %i instead\n',
                i,
                connection_type,
                ConnectionType.LATERAL2FREQ,
                ConnectionType.LATERAL2AMP,
            )


cpdef inline void ode_joints(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    JointsControlArrayCy joints,
    unsigned int n_oscillators,
) nogil:
    """Joints offset

    d_joints_offset = rate*(joints_offset_desired - joints_offset)

    """
    cdef unsigned int joint_i, n_joints = joints.c_n_joints()
    for joint_i in range(n_joints):
        # rate*(joints_offset_desired - joints_offset)
        dstate[2*n_oscillators+joint_i] = joints.c_rate(joint_i)*(
            joints.c_offset_desired(
                iteration,
                joint_i,
                drives,
            ) - joint_offset(state, joint_i, n_oscillators)
        )


cpdef inline DTYPEv1 ode_oscillators_sparse(
    DTYPE time,
    DTYPEv1 state,
    DTYPEv1 dstate,
    unsigned int iteration,
    AmphibiousDataCy data,
    unsigned int nosfb=0,  # No sensory feedback
) nogil:
    """Complete CPG network ODE"""
    ode_dphase(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        oscillators=data.network.oscillators,
        connectivity=data.network.osc2osc_map,
    )
    ode_damplitude(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        oscillators=data.network.oscillators,
    )
    ode_joints(
        iteration=iteration,
        state=state,
        dstate=dstate,
        drives=data.network.drives,
        joints=data.joints,
        n_oscillators=data.network.oscillators.n_oscillators,
    )
    if nosfb:
        return dstate
    ode_stretch(
        iteration=iteration,
        state=state,
        dstate=dstate,
        joints=data.sensors.joints,
        joints2osc_map=data.network.joints2osc_map,
        n_oscillators=data.network.oscillators.n_oscillators,
    )
    ode_contacts(
        iteration=iteration,
        state=state,
        dstate=dstate,
        contacts=data.sensors.contacts,
        contacts2osc_map=data.network.contacts2osc_map,
    )
    ode_xfrc(
        iteration=iteration,
        state=state,
        dstate=dstate,
        xfrc=data.sensors.xfrc,
        xfrc2osc_map=data.network.xfrc2osc_map,
        n_oscillators=data.network.oscillators.n_oscillators,
    )
    return dstate
