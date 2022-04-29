"""Cython controller code"""

include 'types.pxd'
from farms_core.sensors.data_cy cimport (
    JointSensorArrayCy,
    ContactsArrayCy,
    XfrcArrayCy,
)
from ..data.data_cy cimport (
    AmphibiousDataCy,
    NetworkParametersCy,
    DriveArrayCy,
    OscillatorsCy,
    OscillatorsConnectivityCy,
    JointsConnectivityCy,
    ContactsConnectivityCy,
    XfrcConnectivityCy,
    JointsControlArrayCy,
)


cpdef void ode_dphase(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
    OscillatorsConnectivityCy connectivity,
) nogil


cpdef void ode_damplitude(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    OscillatorsCy oscillators,
) nogil


cpdef inline void ode_stretch(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    JointSensorArrayCy joints,
    JointsConnectivityCy joints2osc_map,
    unsigned int n_oscillators,
) nogil


cpdef void ode_contacts(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    ContactsArrayCy contacts,
    ContactsConnectivityCy contacts2osc_map,
) nogil


cpdef void ode_xfrc(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    XfrcArrayCy xfrc,
    XfrcConnectivityCy xfrc2osc_map,
    unsigned int n_oscillators,
) nogil


cpdef void ode_joints(
    unsigned int iteration,
    DTYPEv1 state,
    DTYPEv1 dstate,
    DriveArrayCy drives,
    JointsControlArrayCy joints,
    unsigned int n_oscillators,
) nogil


cpdef DTYPEv1 ode_oscillators_sparse(
    DTYPE time,
    DTYPEv1 state,
    DTYPEv1 dstate,
    unsigned int iteration,
    AmphibiousDataCy data,
    unsigned int nosfb=*,  # No sensory feedback
) nogil
