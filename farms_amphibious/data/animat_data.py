"""Animat data"""

import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
from farms_bullet.data.array import DoubleArray1D
from farms_bullet.data.data import SensorsData
from .animat_data_cy import (
    ConnectionType,
    AnimatDataCy,
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
    JointsArrayCy,
)


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
    'REACTION2FREQ',
    'REACTION2AMP',
    'FRICTION2FREQ',
    'FRICTION2AMP',
    'LATERAL2FREQ',
    'LATERAL2AMP',
]
CONNECTIONTYPE2NAME = dict(zip(ConnectionType, CONNECTIONTYPENAMES))
NAME2CONNECTIONTYPE = dict(zip(CONNECTIONTYPENAMES, ConnectionType))


def connections_from_connectivity(connectivity):
    """Connections from connectivity"""
    return [
        [
            connection['in'],
            connection['out'],
            NAME2CONNECTIONTYPE[connection['type']]
        ]
        for connection in connectivity
    ]


def to_array(array, iteration=None):
    """To array or None"""
    if array is not None:
        array = np.array(array)
        if iteration is not None:
            array = array[:iteration]
    return array


class AnimatData(AnimatDataCy):
    """Animat data"""

    def __init__(self, state=None, network=None, joints=None, sensors=None):
        super(AnimatData, self).__init__()
        self.state = state
        self.network = network
        self.joints = joints
        self.sensors = sensors

    @classmethod
    def from_dict(cls, dictionary, n_oscillators=0):
        """Load data from dictionary"""
        return cls(
            state=OscillatorNetworkState(dictionary['state'], n_oscillators),
            network=NetworkParameters.from_dict(dictionary['network']),
            joints=JointsArray(dictionary['joints']),
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    @classmethod
    def from_file(cls, filename, n_oscillators=0):
        """From file"""
        return cls.from_dict(dd.io.load(filename), n_oscillators)

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        return {
            'state': to_array(self.state.array),
            'network': self.network.to_dict(iteration),
            'joints': to_array(self.joints.array),
            'sensors': self.sensors.to_dict(iteration),
        }

    def to_file(self, filename, iteration=None):
        """Save data to file"""
        dd.io.save(filename, self.to_dict(iteration))

    def plot(self, times):
        """Plot"""
        self.state.plot(times)
        self.sensors.plot(times)


class NetworkParameters(NetworkParametersCy):
    """Network parameter"""

    def __init__(
            self,
            drives,
            oscillators,
            osc_connectivity,
            drive_connectivity,
            joints_connectivity,
            contacts_connectivity,
            hydro_connectivity
    ):
        super(NetworkParameters, self).__init__()
        self.drives = drives
        self.oscillators = oscillators
        self.drive_connectivity = drive_connectivity
        self.joints_connectivity = joints_connectivity
        self.osc_connectivity = osc_connectivity
        self.contacts_connectivity = contacts_connectivity
        self.hydro_connectivity = hydro_connectivity

    @classmethod
    def from_dict(cls, dictionary):
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
        )

    def to_dict(self, iteration=None):
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


class OscillatorNetworkState(OscillatorNetworkStateCy):
    """Network state"""

    def __init__(self, state, n_oscillators):
        super(OscillatorNetworkState, self).__init__(state)
        self.n_oscillators = n_oscillators

    @classmethod
    def from_options(cls, state, animat_options):
        """From options"""
        return cls(
            state=state,
            n_oscillators=2*animat_options.morphology.n_joints()
        )

    @classmethod
    def from_solver(cls, solver, n_oscillators):
        """From solver"""
        return cls(solver.state, n_oscillators, solver.iteration)

    def phases(self, iteration):
        """Phases"""
        return self.array[iteration, :self.n_oscillators]

    def phases_all(self):
        """Phases"""
        return self.array[:, :self.n_oscillators]

    def amplitudes(self, iteration):
        """Amplitudes"""
        return self.array[iteration, self.n_oscillators:]

    def amplitudes_all(self):
        """Phases"""
        return self.array[:, self.n_oscillators:]

    @classmethod
    def from_initial_state(cls, initial_state, n_iterations):
        """From initial state"""
        state_size = len(initial_state)
        state_array = np.zeros([n_iterations, state_size], dtype=NPDTYPE)
        state_array[0, :] = initial_state
        return cls(state_array, n_oscillators=2*state_size//5)

    def plot(self, times):
        """Plot"""
        self.plot_phases(times)
        self.plot_amplitudes(times)
        self.plot_neural_activity_normalised(times)

    def plot_phases(self, times):
        """Plot phases"""
        plt.figure('Network state phases')
        for data in np.transpose(self.phases_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Phases [rad]')
        plt.grid(True)

    def plot_amplitudes(self, times):
        """Plot amplitudes"""
        plt.figure('Network state amplitudes')
        for data in np.transpose(self.amplitudes_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Amplitudes')
        plt.grid(True)

    def plot_phases(self, times):
        """Plot phases"""
        plt.figure('Network state phases')
        for data in np.transpose(self.phases_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Phases [rad]')
        plt.grid(True)

    def plot_neural_activity_normalised(self, times):
        """Plot amplitudes"""
        plt.figure('Neural activities (normalised)')
        for data_i, data in enumerate(np.transpose(self.phases_all())):
            plt.plot(times, 2*data_i + 0.5*(1 + np.cos(data[:len(times)])))
        plt.xlabel('Times [s]')
        plt.ylabel('Neural activity')
        plt.grid(True)


class DriveArray(DriveArrayCy):
    """Drive array"""

    @classmethod
    def from_initial_drive(cls, initial_drives, n_iterations):
        """From initial drive"""
        drive_size = len(initial_drives)
        drive_array = np.zeros([n_iterations, drive_size], dtype=NPDTYPE)
        drive_array[0, :] = initial_drives
        return cls(drive_array)


class DriveDependentArray(DriveDependentArrayCy):
    """Drive dependent array"""

    @classmethod
    def from_vectors(cls, gain, bias, low, high, saturation):
        """From each parameter"""
        return cls(np.array([gain, bias, low, high, saturation]))


class Oscillators(OscillatorsCy):
    """Oscillator array"""

    def __init__(
            self, intrinsic_frequencies, nominal_amplitudes, rates,
            modular_phases, modular_amplitudes,
    ):
        super(Oscillators, self).__init__()
        self.intrinsic_frequencies = DriveDependentArray(intrinsic_frequencies)
        self.nominal_amplitudes = DriveDependentArray(nominal_amplitudes)
        self.rates = DoubleArray1D(rates)
        self.modular_phases = DoubleArray1D(modular_phases)
        self.modular_amplitudes = DoubleArray1D(modular_amplitudes)

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            intrinsic_frequencies=dictionary['intrinsic_frequencies'],
            nominal_amplitudes=dictionary['nominal_amplitudes'],
            rates=dictionary['rates'],
            modular_phases=dictionary['modular_phases'],
            modular_amplitudes=dictionary['modular_amplitudes'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'intrinsic_frequencies': to_array(self.intrinsic_frequencies.array),
            'nominal_amplitudes': to_array(self.nominal_amplitudes.array),
            'rates': to_array(self.rates.array),
            'modular_phases': to_array(self.modular_phases.array),
            'modular_amplitudes': to_array(self.modular_amplitudes.array),
        }

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
            freqs,
            amplitudes,
            np.array(network.osc_rates(), dtype=NPDTYPE),
            np.array(network.osc_modular_phases(), dtype=NPDTYPE),
            np.array(network.osc_modular_amplitudes(), dtype=NPDTYPE),
        )


class OscillatorConnectivity(OscillatorsConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
            desired_phases=dictionary['desired_phases'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
            'desired_phases': to_array(self.desired_phases.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity)
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
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, _iteration=None):
        """Convert data to dictionary"""
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        return cls(
            np.array(connections, dtype=NPUITYPE),
            np.array(weights, dtype=NPDTYPE),
        )


class ContactsConnectivity(ContactsConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, _iteration=None):
        """Convert data to dictionary"""
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        return cls(
            np.array(connections, dtype=NPUITYPE),
            np.array(weights, dtype=NPDTYPE),
        )


class HydroConnectivity(HydroConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        return cls(
            connections=np.array(connections, dtype=NPUITYPE),
            weights=np.array(weights, dtype=NPDTYPE),
        )


class JointsArray(JointsArrayCy):
    """Oscillator array"""

    @classmethod
    def from_options(cls, control):
        """Default"""
        return cls(np.array([
            [
                offset['gain'],
                offset['bias'],
                offset['low'],
                offset['high'],
                offset['saturation'],
                rate,
            ]
            for offset, rate in zip(
                control.joints_offsets(),
                control.joints_rates(),
            )
        ], dtype=NPDTYPE))
