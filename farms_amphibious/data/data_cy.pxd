"""Amphibious data"""

include 'types.pxd'
import numpy as np
cimport numpy as np
from farms_core.sensors.data_cy cimport SensorsDataCy
from farms_core.model.data_cy cimport AnimatDataCy
from farms_core.array.array_cy cimport (
    DoubleArray1D,
    DoubleArray2D,
    IntegerArray1D,
    IntegerArray2D,
)


cpdef enum ConnectionType:
    OSC2OSC
    DRIVE2OSC
    POS2FREQ
    VEL2FREQ
    TOR2FREQ
    POS2AMP
    VEL2AMP
    TOR2AMP
    STRETCH2FREQTEGOTAE
    STRETCH2AMPTEGOTAE
    STRETCH2FREQ
    STRETCH2AMP
    REACTION2FREQ
    REACTION2AMP
    REACTION2FREQTEGOTAE
    FRICTION2FREQ
    FRICTION2AMP
    LATERAL2FREQ
    LATERAL2AMP


cdef class AmphibiousDataCy(AnimatDataCy):
    """Amphibious data"""
    cdef public OscillatorNetworkStateCy state
    cdef public NetworkParametersCy network
    cdef public JointsControlArrayCy joints


cdef class NetworkParametersCy:
    """Network parameters"""
    # Oscillators
    cdef public OscillatorsCy oscillators
    cdef public OscillatorsConnectivityCy osc2osc_map
    # Drives
    cdef public DriveArrayCy drives
    # cdef public IntegerArray1D drive2osc_map
    # Sensors
    cdef public JointsConnectivityCy joints2osc_map
    cdef public ContactsConnectivityCy contacts2osc_map
    cdef public XfrcConnectivityCy xfrc2osc_map


cdef class OscillatorNetworkStateCy(DoubleArray2D):
    """Network state"""
    cdef public unsigned int n_oscillators

    cpdef public DTYPEv1 phases(self, unsigned int iteration)
    cpdef public DTYPEv2 phases_all(self)
    cpdef public DTYPEv1 amplitudes(self, unsigned int iteration)
    cpdef public DTYPEv2 amplitudes_all(self)
    cpdef public DTYPEv1 offsets(self, unsigned int iteration)
    cpdef public DTYPEv2 offsets_all(self)
    cpdef public np.ndarray outputs(self, unsigned int iteration)
    cpdef public np.ndarray outputs_all(self)


cdef class DriveArrayCy(DoubleArray2D):
    """Drive array"""


cdef class DriveDependentArrayCy(DoubleArray2D):
    """Drive dependent array"""
    cdef public unsigned int n_nodes

    cdef inline DTYPE c_gain(self, unsigned int index) nogil:
        """Gain"""
        return self.array[index, 0]

    cdef inline DTYPE c_bias(self, unsigned int index) nogil:
        """Bias"""
        return self.array[index, 1]

    cdef inline DTYPE c_low(self, unsigned int index) nogil:
        """Low"""
        return self.array[index, 2]

    cdef inline DTYPE c_high(self, unsigned int index) nogil:
        """High"""
        return self.array[index, 3]

    cdef inline DTYPE c_saturation(self, unsigned int index) nogil:
        """Saturation"""
        return self.array[index, 4]

    cdef inline DTYPE c_value(self, unsigned int index, DTYPE drive) nogil:
        """Value"""
        return (
            (self.c_gain(index)*drive + self.c_bias(index))
            if self.c_low(index) <= drive <= self.c_high(index)
            else self.c_saturation(index)
        )

    cdef inline DTYPE c_value_mod(self, unsigned int index, DTYPE drive1, DTYPE drive2) nogil:
        """Value"""
        return (
            (self.c_gain(index)*drive2 + self.c_bias(index))
            if self.c_low(index) <= drive1 <= self.c_high(index)
            else self.c_saturation(index)
        )


cdef class OscillatorsCy:
    """Oscillator array"""
    cdef public unsigned int n_oscillators
    cdef public IntegerArray1D drive2osc_map
    cdef public DriveDependentArrayCy intrinsic_frequencies
    cdef public DriveDependentArrayCy nominal_amplitudes
    cdef public DoubleArray1D rates
    cdef public DoubleArray1D modular_phases
    cdef public DoubleArray1D modular_amplitudes

    cdef inline DTYPE c_angular_frequency(
        self,
        unsigned int iteration,
        unsigned int index,
        DriveArrayCy drives,
    ) nogil:
        """Angular frequency"""
        cdef unsigned int drive_index = self.drive2osc_map.array[index]
        return self.intrinsic_frequencies.c_value(
            index,
            drives.array[iteration, drive_index],
        )

    cdef inline DTYPE c_nominal_amplitude(
        self,
        unsigned int iteration,
        unsigned int index,
        DriveArrayCy drives,
    ) nogil:
        """Nominal amplitude"""
        cdef unsigned int drive_index = self.drive2osc_map.array[index]
        return self.nominal_amplitudes.c_value(
            index,
            drives.array[iteration, drive_index],
        )

    cdef inline DTYPE c_rate(self, unsigned int index) nogil:
        """Rate"""
        return self.rates.array[index]

    cdef inline DTYPE c_modular_phases(self, unsigned int index) nogil:
        """Modular phase"""
        return self.modular_phases.array[index]

    cdef inline DTYPE c_modular_amplitudes(self, unsigned int index) nogil:
        """Modular amplitude"""
        return self.modular_amplitudes.array[index]


cdef class ConnectivityCy:
    """Connectivity array"""
    cdef public unsigned int n_connections
    cdef readonly IntegerArray2D connections

    cpdef UITYPE input(self, unsigned int connection_i)
    cpdef UITYPE output(self, unsigned int connection_i)
    cpdef UITYPE connection_type(self, unsigned int connection_i)


cdef class OscillatorsConnectivityCy(ConnectivityCy):
    """oscillator connectivity array"""
    cdef readonly DoubleArray1D weights
    cdef readonly DoubleArray1D desired_phases

    cdef inline DTYPE c_weight(self, unsigned int index) nogil:
        """Weight"""
        return self.weights.array[index]

    cdef inline DTYPE c_desired_phase(self, unsigned int index) nogil:
        """Desired phase"""
        return self.desired_phases.array[index]


cdef class JointsConnectivityCy(ConnectivityCy):
    """Joint connectivity array"""
    cdef readonly DoubleArray1D weights

    cdef inline DTYPE c_weight(self, unsigned int index) nogil:
        """Weight"""
        return self.weights.array[index]


cdef class ContactsConnectivityCy(ConnectivityCy):
    """Contact connectivity array"""
    cdef readonly DoubleArray1D weights

    cdef inline DTYPE c_weight(self, unsigned int index) nogil:
        """Weight"""
        return self.weights.array[index]


cdef class XfrcConnectivityCy(ConnectivityCy):
    """External forces connectivity array"""
    cdef readonly DoubleArray1D weights

    cdef inline DTYPE c_weights(self, unsigned int index) nogil:
        """Weight for external forces frequency"""
        return self.weights.array[index]


cdef class JointsControlArrayCy(DriveDependentArrayCy):
    """Joints control array"""
    cdef public IntegerArray2D drive2joint_map

    cdef inline unsigned int c_n_joints(self) nogil:
        """Number of joints"""
        return self.n_nodes

    cdef inline DTYPE c_offset_desired(
        self,
        unsigned int iteration,
        unsigned int index,
        DriveArrayCy drives,
    ) nogil:
        """Desired offset"""
        cdef double drive0 = drives.array[iteration, self.drive2joint_map.array[index, 0]]
        cdef double drive1 = drives.array[iteration, self.drive2joint_map.array[index, 1]]
        return self.c_value_mod(
            index,
            0.5*(drive0+drive1),
            drive1-drive0,
        )

    cdef inline DTYPE c_rate(self, unsigned int index) nogil:
        """Rate"""
        return self.array[index, 5]
