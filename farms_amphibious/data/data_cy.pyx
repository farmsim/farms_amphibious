"""Amphibious data"""

from typing import Any
import numpy as np
cimport numpy as np
from nptyping import NDArray, Shape


cdef class AmphibiousDataCy(AnimatDataCy):
    """Amphibious data"""
    pass


cdef class NetworkParametersCy:
    """Network parameters"""

    def __init__(
            self,
            drives: DriveArrayCy,
            # drive2osc_map: ConnectivityCy,
            oscillators: OscillatorsCy,
            osc2osc_map: OscillatorsConnectivityCy,
            joints2osc_map: JointsConnectivityCy,
            contacts2osc_map: ContactsConnectivityCy,
            xfrc2osc_map: XfrcConnectivityCy,
    ):
        super().__init__()
        self.drives = drives
        self.oscillators = oscillators
        # self.drive2osc_map = drive2osc_map
        self.joints2osc_map = joints2osc_map
        self.osc2osc_map = osc2osc_map
        self.contacts2osc_map = contacts2osc_map
        self.xfrc2osc_map = xfrc2osc_map


cdef class OscillatorNetworkStateCy(DoubleArray2D):
    """Oscillator network state"""

    def __init__(
            self,
            array: NDArray[Shape['*, *'], np.double],
            n_oscillators: int,
    ):
        assert np.ndim(array) == 2, 'Ndim {np.ndim(array)} != 2'
        assert n_oscillators > 1, f'n_oscillators={n_oscillators} must be > 1'
        super().__init__(array=array)
        self.n_oscillators = n_oscillators

    cpdef DTYPEv1 phases(self, unsigned int iteration):
        """Oscillators phases"""
        return self.array[iteration, :self.n_oscillators]

    cpdef DTYPEv2 phases_all(self):
        """Oscillators phases"""
        return self.array[:, :self.n_oscillators]

    cpdef DTYPEv1 amplitudes(self, unsigned int iteration):
        """Amplitudes"""
        return self.array[iteration, self.n_oscillators:2*self.n_oscillators]

    cpdef DTYPEv2 amplitudes_all(self):
        """Amplitudes"""
        return self.array[:, self.n_oscillators:2*self.n_oscillators]

    cpdef DTYPEv1 offsets(self, unsigned int iteration):
        """Offset"""
        return self.array[iteration, 2*self.n_oscillators:]

    cpdef DTYPEv2 offsets_all(self):
        """Offset"""
        return self.array[:, 2*self.n_oscillators:]

    cpdef np.ndarray outputs(self, unsigned int iteration):
        """Outputs"""
        return self.amplitudes(iteration)*(1 + np.cos(self.phases(iteration)))

    cpdef np.ndarray outputs_all(self):
        """Outputs"""
        return self.amplitudes_all()*(1 + np.cos(self.phases_all()))


cdef class DriveArrayCy(DoubleArray2D):
    """Drive array"""

    def __init__(
            self,
            array: NDArray[(Any, Any), np.double],
            left_indices: NDArray[(Any,), np.uintc],
            right_indices: NDArray[(Any,), np.uintc],
    ):
        super().__init__(array=array)
        self.left_indices = np.array(left_indices, dtype=np.uintc)
        self.right_indices = np.array(right_indices, dtype=np.uintc)


cdef class DriveDependentArrayCy(DoubleArray2D):
    """Drive dependent array"""

    def __init__(
            self,
            array: NDArray[(Any, Any), np.double],
    ):
        super().__init__(array=array)
        self.n_nodes = np.shape(array)[0]


cdef class OscillatorsCy:
    """Oscillators"""

    def __init__(
            self,
            n_oscillators: int,
            drive2osc_map: NDArray[Shape['*'], np.uint],
            intrinsic_frequencies: DriveDependentArrayCy,
            nominal_amplitudes: DriveDependentArrayCy,
            rates: NDArray[Shape['*'], np.double],
            modular_phases: NDArray[Shape['*'], np.double],
            modular_amplitudes: NDArray[Shape['*'], np.double],
    ):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.drive2osc_map = drive2osc_map
        self.intrinsic_frequencies = intrinsic_frequencies
        self.nominal_amplitudes = nominal_amplitudes
        self.rates = rates
        self.modular_phases = modular_phases
        self.modular_amplitudes = modular_amplitudes


cdef class ConnectivityCy:
    """Connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
    ):
        super(ConnectivityCy, self).__init__()
        if connections is not None and list(connections):
            shape = np.shape(connections)
            assert shape[1] == 3, (
                f'Connections should be of dim 3, got {shape[1]}'
            )
            self.n_connections = shape[0]
            self.connections = IntegerArray2D(connections)
        else:
            self.n_connections = 0
            self.connections = IntegerArray2D(None)

    cpdef UITYPE input(self, unsigned int connection_i):
        """Node input"""
        self.array[connection_i, 0]

    cpdef UITYPE output(self, unsigned int connection_i):
        """Node output"""
        self.array[connection_i, 1]

    cpdef UITYPE connection_type(self, unsigned int connection_i):
        """Connection type"""
        self.array[connection_i, 2]


cdef class OscillatorsConnectivityCy(ConnectivityCy):
    """Oscillators connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
            weights: NDArray[(Any,), np.double],
            desired_phases: NDArray[(Any,), np.double],
    ):
        super(OscillatorsConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                f'Size of connections {size}'
                f' != size of size of weights {len(weights)}'
            )
            assert size == len(desired_phases), (
                f'Size of connections {size}'
                f' != size of size of phases {len(desired_phases)}'
            )
            self.weights = DoubleArray1D(weights)
            self.desired_phases = DoubleArray1D(desired_phases)
        else:
            self.weights = DoubleArray1D(None)
            self.desired_phases = DoubleArray1D(None)


cdef class JointsConnectivityCy(ConnectivityCy):
    """Joints connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
            weights: NDArray[(Any,), np.double],
    ):
        super(JointsConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                f'Size of connections {size}'
                f' != size of size of weights {len(weights)}'
            )
            self.weights = DoubleArray1D(weights)
        else:
            self.weights = DoubleArray1D(None)


cdef class ContactsConnectivityCy(ConnectivityCy):
    """Contacts connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
            weights: NDArray[(Any,), np.double],
    ):
        super(ContactsConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                f'Size of connections {size}'
                f' != size of size of weights {len(weights)}'
            )
            self.weights = DoubleArray1D(weights)
        else:
            self.weights = DoubleArray1D(None)


cdef class XfrcConnectivityCy(ConnectivityCy):
    """External forces connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
            weights: NDArray[(Any,), np.double],
    ):
        super(XfrcConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                f'Size of connections {size}'
                f' != size of size of weights {len(weights)}'
            )
            self.weights = DoubleArray1D(weights)
        else:
            self.weights = DoubleArray1D(None)


cdef class JointsControlArrayCy(DriveDependentArrayCy):
    """Joints control array"""

    def __init__(
            self,
            array: NDArray[(Any, Any), np.double],
            drive2joint_map: NDArray[(Any, Any), np.uint],
    ):
        super().__init__(array=array)
        self.drive2joint_map = drive2joint_map
