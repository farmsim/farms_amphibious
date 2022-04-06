"""Network"""

from typing import List, Tuple, Dict, Any

import numpy as np
from nptyping import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from farms_core.array.array import to_array
from farms_core.array.array_cy import DoubleArray1D

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
    HydroConnectivityCy,
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
            initial_state: NDArray[(Any,), float],
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

    def plot(self, times: NDArray[(Any,), float]) -> Dict:
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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

    @classmethod
    def from_initial_drive(
            cls,
            initial_drives: List[float],
            n_iterations: int,
    ):
        """From initial drive"""
        drive_size = len(initial_drives)
        drive_array = np.full(
            shape=[n_iterations, drive_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        drive_array[0, :] = initial_drives
        return cls(drive_array)

    def plot(
            self,
            times: NDArray[(Any,), float],
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
            intrinsic_frequencies: NDArray[(Any, Any), np.double],
            nominal_amplitudes: NDArray[(Any, Any), np.double],
            rates: NDArray[(Any,), np.double],
            modular_phases: NDArray[(Any,), np.double],
            modular_amplitudes: NDArray[(Any,), np.double],
    ):
        super().__init__(n_oscillators=len(names))
        self.names = names
        self.intrinsic_frequencies = DriveDependentArray(intrinsic_frequencies)
        self.nominal_amplitudes = DriveDependentArray(nominal_amplitudes)
        self.rates = DoubleArray1D(rates)
        self.modular_phases = DoubleArray1D(modular_phases)
        self.modular_amplitudes = DoubleArray1D(modular_amplitudes)

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


class HydroConnectivity(HydroConnectivityCy):
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

    def __init__(
            self,
            drives: DriveArray,
            oscillators: Oscillators,
            osc_connectivity: OscillatorConnectivity,
            drive_connectivity: ConnectivityCy,
            joints_connectivity: JointsConnectivity,
            contacts_connectivity: ContactsConnectivity,
            hydro_connectivity: HydroConnectivity,
    ):
        super().__init__()
        self.drives = drives
        self.oscillators = oscillators
        self.drive_connectivity = drive_connectivity
        self.joints_connectivity = joints_connectivity
        self.osc_connectivity = osc_connectivity
        self.contacts_connectivity = contacts_connectivity
        self.hydro_connectivity = hydro_connectivity

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            drives=DriveArray(
                dictionary['drives']
            ),
            oscillators=Oscillators.from_dict(
                dictionary['oscillators']
            ),
            osc_connectivity=OscillatorConnectivity.from_dict(
                dictionary['osc_connectivity']
            ),
            drive_connectivity=ConnectivityCy(
                dictionary['drive_connectivity']
            ),
            joints_connectivity=JointsConnectivity.from_dict(
                dictionary['joints_connectivity']
            ),
            contacts_connectivity=ContactsConnectivity.from_dict(
                dictionary['contacts_connectivity']
            ),
            hydro_connectivity=HydroConnectivity.from_dict(
                dictionary['hydro_connectivity']
            ),
        ) if dictionary else None

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'drives': to_array(self.drives.array),
            'oscillators': self.oscillators.to_dict(),
            'osc_connectivity': self.osc_connectivity.to_dict(),
            'drive_connectivity': self.drive_connectivity.connections.array,
            'joints_connectivity': self.joints_connectivity.to_dict(),
            'contacts_connectivity': self.contacts_connectivity.to_dict(),
            'hydro_connectivity': self.hydro_connectivity.to_dict(),
        }
