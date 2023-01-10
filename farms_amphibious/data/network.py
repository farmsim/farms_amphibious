"""Network"""

from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from farms_core.array.array import to_array
from farms_core.array.array_cy import (
    IntegerArray1D,
    DoubleArray1D,
    DoubleArray2D,
)
from farms_core.array.types import (
    NDARRAY_V1,
    NDARRAY_V1_I,
    NDARRAY_V1_D,
    NDARRAY_V2_D,
)

from .data_cy import (
    ConnectionType,
    NetworkParametersCy,
    OscillatorNetworkStateCy,
    DriveArrayCy,
    DriveDependentArrayCy,
    OscillatorsCy,
    ConnectivityCy,
    OscillatorsConnectivityCy,
    JointsConnectivityCy,
    ContactsConnectivityCy,
    XfrcConnectivityCy,
)

# pylint: disable=no-member


NPDTYPE = np.float64
NPUITYPE = np.uintc


CONNECTIONTYPENAMES = [
    'OSC2OSC',
    'DRIVE2OSC',
    'POS2FREQ',
    'VEL2FREQ',
    'TOR2FREQ',
    'POS2AMP',
    'VEL2AMP',
    'TOR2AMP',
    'STRETCH2FREQTEGOTAE',
    'STRETCH2AMPTEGOTAE',
    'STRETCH2FREQ',
    'STRETCH2AMP',
    'REACTION2FREQ',
    'REACTION2AMP',
    'REACTION2FREQTEGOTAE',
    'FRICTION2FREQ',
    'FRICTION2AMP',
    'LATERAL2FREQ',
    'LATERAL2AMP',
]
assert len(ConnectionType) == len(CONNECTIONTYPENAMES)
CONNECTIONTYPE2NAME = dict(zip(ConnectionType, CONNECTIONTYPENAMES))
NAME2CONNECTIONTYPE = dict(zip(CONNECTIONTYPENAMES, ConnectionType))


def connections_from_connectivity(
        connectivity: List[Dict],
        map1: Dict = None,
        map2: Dict = None,
) -> List[Tuple[int, int, int]]:
    """Connections from connectivity"""
    if map1 or map2:
        for connection in connectivity:
            if map1:
                assert connection['in'] in map1, (
                    f'Connection {connection["in"]} not in map {map1}'
                )
            if map2:
                assert connection['out'] in map2, (
                    f'Connection {connection["out"]} not in map {map2}'
                )
    return [
        [
            map1[connection['in']] if map1 else connection['in'],
            map2[connection['out']] if map2 else connection['out'],
            NAME2CONNECTIONTYPE[connection['type']]
        ]
        for connection in connectivity
    ]


class OscillatorNetworkState(OscillatorNetworkStateCy):
    """Network state"""

    @classmethod
    def from_initial_state(
            cls,
            initial_state: NDARRAY_V1,
            n_iterations: int,
            n_oscillators: int,
    ):
        """From initial state"""
        state_size = len(initial_state)
        state_array = np.full(
            shape=[n_iterations, state_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        state_array[0, :] = initial_state
        return cls(array=state_array, n_oscillators=n_oscillators)

    def plot(self, times: NDARRAY_V1) -> Dict:
        """Plot"""
        return {
            'phases': self.plot_phases(times),
            'amplitudes': self.plot_amplitudes(times),
            'neural_activity_normalised': (
                self.plot_neural_activity_normalised(times)
            ),
        }

    def plot_phases(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot phases"""
        fig = plt.figure('Network state phases')
        for data in np.transpose(self.phases_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Phases [rad]')
        plt.grid(True)
        return fig

    def plot_amplitudes(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot amplitudes"""
        fig = plt.figure('Network state amplitudes')
        for data in np.transpose(self.amplitudes_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Amplitudes')
        plt.grid(True)
        return fig

    def plot_neural_activity_normalised(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot amplitudes"""
        fig = plt.figure('Neural activities (normalised)')
        for data_i, data in enumerate(np.transpose(self.phases_all())):
            plt.plot(times, 2*data_i + 0.5*(1 + np.cos(data[:len(times)])))
        plt.xlabel('Times [s]')
        plt.ylabel('Neural activity')
        plt.grid(True)
        return fig


class DriveArray(DriveArrayCy):
    """Drive array"""

    def __init__(
            self,
            array: NDARRAY_V2_D,
            left_indices: NDARRAY_V1_I,
            right_indices: NDARRAY_V1_I,
            contacts_indices: List[List[int]] = None,
    ):
        super().__init__(
            array=array,
            left_indices=left_indices,
            right_indices=right_indices,
        )
        self.contacts_indices = contacts_indices

    @classmethod
    def from_animat_options(
            cls,
            animat_options: AmphibiousOptions,
            n_iterations: int,
    ):
        """From initial drive"""
        control = animat_options.control
        initial_drives = control.network.drives_init()
        drive_size = len(initial_drives)
        drive_array = np.full(
            shape=[n_iterations, drive_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        drive_array[0, :] = initial_drives
        return cls(
            array=drive_array,
            left_indices=control.network.drives_left_indices(),
            right_indices=control.network.drives_right_indices(),
            contacts_indices=control.drives_contacts_indices(),
        )

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        contacts_indices = [
            [index for index in indices if not index < 0]
            for indices in dictionary['contacts_indices']
        ]
        return cls(
            array=dictionary['array'],
            left_indices=dictionary['left_indices'],
            right_indices=dictionary['right_indices'],
            contacts_indices=contacts_indices,
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        contacts_indices = self.contacts_indices
        if contacts_indices:
            maxlen = max(len(indices) for indices in contacts_indices)
            contacts_indices = np.full(
                shape=[len(contacts_indices), maxlen],
                fill_value=-1,
            )
            positives =  np.arange(self.array.shape[0])
            for indices_i, indices in enumerate(self.contacts_indices):
                for index_i, index in enumerate(indices):
                    contacts_indices[indices_i, index_i] = positives[index]
        return {
            'array': self.array,
            'left_indices': to_array(self.left_indices),
            'right_indices': to_array(self.right_indices),
            'contacts_indices': contacts_indices,
        }

    def plot(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot phases"""
        fig = plt.figure('Drives')
        for i, data in enumerate(np.transpose(np.array(self.array))):
            plt.plot(times, data[:len(times)], label=i)
        plt.xlabel('Times [s]')
        plt.ylabel('Drive value')
        plt.grid(True)
        plt.legend()
        return fig


class DriveDependentArray(DriveDependentArrayCy):
    """Drive dependent array"""

    @classmethod
    def from_vectors(
            cls,
            gain: float,
            bias: float,
            low: float,
            high: float,
            saturation: float,
    ):
        """From each parameter"""
        return cls(np.array([gain, bias, low, high, saturation]))


class Oscillators(OscillatorsCy):
    """Oscillator array"""

    def __init__(
            self,
            names: List[str],
            drive2osc_map: NDARRAY_V1_I,
            intrinsic_frequencies: NDARRAY_V2_D,
            nominal_amplitudes: NDARRAY_V2_D,
            rates: NDARRAY_V1_D,
            modular_phases: NDARRAY_V1_D,
            modular_amplitudes: NDARRAY_V1_D,
    ):
        super().__init__(
            n_oscillators=len(names),
            drive2osc_map=IntegerArray1D(drive2osc_map),
            intrinsic_frequencies=DriveDependentArray(intrinsic_frequencies),
            nominal_amplitudes=DriveDependentArray(nominal_amplitudes),
            rates=DoubleArray1D(rates),
            modular_phases=DoubleArray1D(modular_phases),
            modular_amplitudes=DoubleArray1D(modular_amplitudes),
        )
        self.names = names

    @classmethod
    def from_options(cls, network):
        """Default"""
        freqs, amplitudes = [
            np.array([
                [
                    freq['gain'],
                    freq['bias'],
                    freq['low'],
                    freq['high'],
                    freq['saturation'],
                ]
                for freq in option
            ], dtype=NPDTYPE)
            for option in [network.osc_frequencies(), network.osc_amplitudes()]
        ]
        return cls(
            network.osc_names(),
            np.array(network.drive2osc, dtype=NPUITYPE),
            freqs,
            amplitudes,
            np.array(network.osc_rates(), dtype=NPDTYPE),
            np.array(network.osc_modular_phases(), dtype=NPDTYPE),
            np.array(network.osc_modular_amplitudes(), dtype=NPDTYPE),
        )

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            names=dictionary['names'],
            drive2osc_map=dictionary['drive2osc_map'],
            intrinsic_frequencies=dictionary['intrinsic_frequencies'],
            nominal_amplitudes=dictionary['nominal_amplitudes'],
            rates=dictionary['rates'],
            modular_phases=dictionary['modular_phases'],
            modular_amplitudes=dictionary['modular_amplitudes'],
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'names': self.names,
            'drive2osc_map': to_array(self.drive2osc_map.array),
            'intrinsic_frequencies': to_array(self.intrinsic_frequencies.array),
            'nominal_amplitudes': to_array(self.nominal_amplitudes.array),
            'rates': to_array(self.rates.array),
            'modular_phases': to_array(self.modular_phases.array),
            'modular_amplitudes': to_array(self.modular_amplitudes.array),
        }


class OscillatorConnectivity(OscillatorsConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
            desired_phases=dictionary['desired_phases'],
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
            'desired_phases': to_array(self.desired_phases.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity: List[Dict], **kwargs):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity, **kwargs)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        phase_bias = [
            connection['phase_bias']
            for connection in connectivity
        ]
        return cls(
            connections=np.array(connections, dtype=NPUITYPE),
            weights=np.array(weights, dtype=NPDTYPE),
            desired_phases=np.array(phase_bias, dtype=NPDTYPE),
        )


class JointsConnectivity(JointsConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_connectivity(cls, connectivity: List[Dict], **kwargs):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity, **kwargs)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        return cls(
            connections=np.array(connections, dtype=NPUITYPE),
            weights=np.array(weights, dtype=NPDTYPE),
        )

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, _iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }


class ContactsConnectivity(ContactsConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_connectivity(cls, connectivity: List[Dict], **kwargs):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity, **kwargs)
        assert all(
            isinstance(val, int)
            for conn in connections
            for val in conn
        ), f'All connections must be integers:\n{connections}'
        weights = [connection['weight'] for connection in connectivity]
        return cls(
            connections=np.array(connections, dtype=NPUITYPE),
            weights=np.array(weights, dtype=NPDTYPE),
        )

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, _iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }


class XfrcConnectivity(XfrcConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_connectivity(cls, connectivity: List[Dict], **kwargs):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity, **kwargs)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        return cls(
            connections=np.array(connections, dtype=NPUITYPE),
            weights=np.array(weights, dtype=NPDTYPE),
        )

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }


class NetworkParameters(NetworkParametersCy):
    """Network parameter"""

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            drives=DriveArray.from_dict(
                dictionary['drives']
            ),
            oscillators=Oscillators.from_dict(
                dictionary['oscillators']
            ),
            osc2osc_map=OscillatorConnectivity.from_dict(
                dictionary['osc2osc_map']
            ),
            joints2osc_map=JointsConnectivity.from_dict(
                dictionary['joints2osc_map']
            ),
            contacts2osc_map=ContactsConnectivity.from_dict(
                dictionary['contacts2osc_map']
            ),
            xfrc2osc_map=XfrcConnectivity.from_dict(
                dictionary['xfrc2osc_map']
            ),
        ) if dictionary else None

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'drives': self.drives.to_dict(),
            'oscillators': self.oscillators.to_dict(),
            'osc2osc_map': self.osc2osc_map.to_dict(),
            'joints2osc_map': self.joints2osc_map.to_dict(),
            'contacts2osc_map': self.contacts2osc_map.to_dict(),
            'xfrc2osc_map': self.xfrc2osc_map.to_dict(),
        }
