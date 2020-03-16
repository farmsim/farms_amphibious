"""Animat data"""

import sys
import numpy as np

from .convention import AmphibiousConvention
from farms_amphibious.data.animat_data import (
    OscillatorNetworkState,
    AnimatData,
    NetworkParameters,
    OscillatorArray,
    ConnectivityArray,
    JointsArray,
    SensorsData,
    ContactsArray,
    ProprioceptionArray,
    GpsArray,
    HydrodynamicsArray
)


class AmphibiousOscillatorNetworkState(OscillatorNetworkState):
    """Network state"""

    @staticmethod
    def default_initial_state(options):
        """Default state"""
        n_joints = options.morphology.n_joints()
        return 1e-3*np.arange(5*n_joints) + np.concatenate([
            # 0*np.linspace(2*np.pi, 0, n_joints),
            np.zeros(n_joints),
            np.zeros(n_joints),
            np.zeros(2*n_joints),
            np.zeros(n_joints)
        ])

    @staticmethod
    def default_state(n_iterations, options):
        """Default state"""
        n_joints = options.morphology.n_joints()
        n_oscillators = 2*n_joints
        return AmphibiousOscillatorNetworkState.from_initial_state(
            initial_state=(
                AmphibiousOscillatorNetworkState.default_initial_state(options)
            ),
            n_iterations=n_iterations,
            n_oscillators=n_oscillators
        )

    @classmethod
    def from_initial_state(cls, initial_state, n_iterations, n_oscillators):
        """From initial state"""
        state = np.zeros(
            [n_iterations, 2, np.shape(initial_state)[0]],
            dtype=np.float64
        )
        state[0, 0, :] = np.array(initial_state)
        return cls(state, n_oscillators)


class AmphibiousData(AnimatData):
    """Amphibious network parameter"""

    @classmethod
    def from_options(cls, state, options, n_iterations):
        """Default amphibious newtwork parameters"""
        oscillators = AmphibiousOscillatorArray.from_options(options)
        connectivity = AmphibiousOscillatorConnectivityArray.from_options(options)
        contacts_connectivity = AmphibiousContactsConnectivityArray.from_options(
            options
        )
        hydro_connectivity = AmphibiousHydroConnectivityArray.from_options(
            options
        )
        network = NetworkParameters(
            oscillators,
            connectivity,
            contacts_connectivity,
            hydro_connectivity
        )
        joints = AmphibiousJointsArray.from_options(options)
        contacts = AmphibiousContactsArray.from_options(options, n_iterations)
        proprioception = AmphibiousProprioceptionArray.from_options(
            options,
            n_iterations
        )
        gps = AmphibiousGpsArray.from_options(
            options,
            n_iterations
        )
        hydrodynamics = AmphibiousHydrodynamicsArray.from_options(
            options,
            n_iterations
        )
        sensors = SensorsData(contacts, proprioception, gps, hydrodynamics)
        return cls(state, network, joints, sensors)


class AmphibiousOscillatorArray(OscillatorArray):
    """Oscillator array"""

    @staticmethod
    def set_options(options):
        """Walking parameters"""
        osc_options = options.control.network.oscillators
        drives = options.control.drives
        n_body = options.morphology.n_joints_body
        n_dof_legs = options.morphology.n_dof_legs
        n_legs = options.morphology.n_legs
        # n_oscillators = 2*(options.morphology.n_joints_body)
        n_oscillators = 2*(options.morphology.n_joints())
        freqs_body = 2*np.pi*np.ones(2*options.morphology.n_joints_body)*(
            osc_options.body_freqs.value(drives)
        )
        freqs_legs = 2*np.pi*np.ones(2*options.morphology.n_joints_legs())*(
            osc_options.legs_freqs.value(drives)
        )
        freqs = np.concatenate([freqs_body, freqs_legs])
        rates = 10*np.ones(n_oscillators)
        # Amplitudes
        amplitudes = np.zeros(n_oscillators)
        for i in range(n_body):
            # amplitudes[[i, i+n_body]] = 0.1+0.2*i/(n_body-1)
            amplitudes[[i, i+n_body]] = (
                osc_options.body_nominal_amplitudes[i].value(drives)
            )
            # osc_options.body_stand_amplitude*np.sin(
            #     2*np.pi*i/n_body
            #     - osc_options.body_stand_shift
            # )
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                amplitudes[[
                    2*n_body + 2*leg_i*n_dof_legs + i,
                    2*n_body + 2*leg_i*n_dof_legs + i + n_dof_legs
                ]] = osc_options.legs_nominal_amplitudes[i].value(drives)
        # print("Amplitudes along body: abs({})".format(amplitudes[:11]))
        return np.abs(freqs), np.abs(rates), np.abs(amplitudes)

    @classmethod
    def from_options(cls, options):
        """Default"""
        freqs, rates, amplitudes = cls.set_options(options)
        return cls.from_parameters(freqs, rates, amplitudes)

    def update(self, options):
        """Update from options

        :param options: Animat options

        """
        freqs, _, amplitudes = self.set_options(options)
        self.freqs[:] = freqs
        self.amplitudes_desired[:] = amplitudes


class AmphibiousOscillatorConnectivityArray(ConnectivityArray):
    """Connectivity array"""

    @staticmethod
    def set_options(options, verbose=False):
        """Walking parameters"""
        # osc_options = options.control.network.oscillators
        conn_options = options.control.network.connectivity
        n_body_joints = options.morphology.n_joints_body
        connectivity = []
        body_amplitude = conn_options.weight_osc_body
        legs_amplitude_internal = conn_options.weight_osc_legs_internal
        legs_amplitude_opposite = conn_options.weight_osc_legs_opposite
        legs_amplitude_following = conn_options.weight_osc_legs_following
        legs2body_amplitude = conn_options.weight_osc_legs2body

        # # Amplitudes
        # amplitudes = [
        #     osc_options.body_stand_amplitude*np.sin(
        #         2*np.pi*i/n_body_joints
        #         - osc_options.body_stand_shift
        #     )
        #     for i in range(n_body_joints)
        # ]

        # Body
        convention = AmphibiousConvention(options)
        for i in range(n_body_joints):
            # i - i
            connectivity.append([
                convention.bodyosc2index(joint_i=i, side=1),
                convention.bodyosc2index(joint_i=i, side=0),
                body_amplitude, np.pi
            ])
            connectivity.append([
                convention.bodyosc2index(joint_i=i, side=0),
                convention.bodyosc2index(joint_i=i, side=1),
                body_amplitude, np.pi
            ])
        for i in range(n_body_joints-1):
            # i - i+1
            phase_diff = options.control.network.connectivity.body_phase_bias
            phase_follow = options.control.network.connectivity.leg_phase_follow
            # phase_diff = np.pi/11
            for side in range(2):
                connectivity.append([
                    convention.bodyosc2index(joint_i=i+1, side=side),
                    convention.bodyosc2index(joint_i=i, side=side),
                    body_amplitude, phase_diff
                ])
                connectivity.append([
                    convention.bodyosc2index(joint_i=i, side=side),
                    convention.bodyosc2index(joint_i=i+1, side=side),
                    body_amplitude, -phase_diff
                ])

        # Legs (internal)
        for leg_i in range(options.morphology.n_legs//2):
            for side_i in range(2):
                _options = {
                    "leg_i": leg_i,
                    "side_i": side_i
                }
                # 0 - 0
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=0, side=1),
                    convention.legosc2index(**_options, joint_i=0, side=0),
                    legs_amplitude_internal, np.pi
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=0, side=0),
                    convention.legosc2index(**_options, joint_i=0, side=1),
                    legs_amplitude_internal, np.pi
                ])
                # 0 - 1
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=1, side=0),
                    convention.legosc2index(**_options, joint_i=0, side=0),
                    legs_amplitude_internal, 0.5*np.pi
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=0, side=0),
                    convention.legosc2index(**_options, joint_i=1, side=0),
                    legs_amplitude_internal, -0.5*np.pi
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=1, side=1),
                    convention.legosc2index(**_options, joint_i=0, side=1),
                    legs_amplitude_internal, 0.5*np.pi
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=0, side=1),
                    convention.legosc2index(**_options, joint_i=1, side=1),
                    legs_amplitude_internal, -0.5*np.pi
                ])
                # 1 - 1
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=1, side=1),
                    convention.legosc2index(**_options, joint_i=1, side=0),
                    legs_amplitude_internal, np.pi
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=1, side=0),
                    convention.legosc2index(**_options, joint_i=1, side=1),
                    legs_amplitude_internal, np.pi
                ])
                # 0 - 2
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=2, side=0),
                    convention.legosc2index(**_options, joint_i=0, side=0),
                    legs_amplitude_internal, 0
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=0, side=0),
                    convention.legosc2index(**_options, joint_i=2, side=0),
                    legs_amplitude_internal, 0
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=2, side=1),
                    convention.legosc2index(**_options, joint_i=0, side=1),
                    legs_amplitude_internal, 0
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=0, side=1),
                    convention.legosc2index(**_options, joint_i=2, side=1),
                    legs_amplitude_internal, 0
                ])
                # 2 - 2
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=2, side=1),
                    convention.legosc2index(**_options, joint_i=2, side=0),
                    legs_amplitude_internal, np.pi
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=2, side=0),
                    convention.legosc2index(**_options, joint_i=2, side=1),
                    legs_amplitude_internal, np.pi
                ])
                # 1 - 3
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=3, side=0),
                    convention.legosc2index(**_options, joint_i=1, side=0),
                    legs_amplitude_internal, 0
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=1, side=0),
                    convention.legosc2index(**_options, joint_i=3, side=0),
                    legs_amplitude_internal, 0
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=3, side=1),
                    convention.legosc2index(**_options, joint_i=1, side=1),
                    legs_amplitude_internal, 0
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=1, side=1),
                    convention.legosc2index(**_options, joint_i=3, side=1),
                    legs_amplitude_internal, 0
                ])
                # 3 - 3
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=3, side=1),
                    convention.legosc2index(**_options, joint_i=3, side=0),
                    legs_amplitude_internal, np.pi
                ])
                connectivity.append([
                    convention.legosc2index(**_options, joint_i=3, side=0),
                    convention.legosc2index(**_options, joint_i=3, side=1),
                    legs_amplitude_internal, np.pi
                ])

        # Opposite leg interaction
        for leg_i in range(options.morphology.n_legs//2):
            for joint_i in range(options.morphology.n_dof_legs):
                for side in range(2):
                    _options = {
                        "joint_i": joint_i,
                        "side": side
                    }
                    connectivity.append([
                        convention.legosc2index(
                            leg_i=leg_i, side_i=0, **_options
                        ),
                        convention.legosc2index(
                            leg_i=leg_i, side_i=1, **_options
                        ),
                        legs_amplitude_opposite, np.pi
                    ])
                    connectivity.append([
                        convention.legosc2index(
                            leg_i=leg_i, side_i=1, **_options
                        ),
                        convention.legosc2index(
                            leg_i=leg_i, side_i=0, **_options
                        ),
                        legs_amplitude_opposite, np.pi
                    ])

        # Following leg interaction
        for leg_pre in range(options.morphology.n_legs//2-1):
            for side_i in range(2):
                for side in range(2):
                    _options = {
                        "side_i": side_i,
                        "side": side
                    }
                    connectivity.append([
                        convention.legosc2index(
                            leg_i=leg_pre,
                            joint_i=0,
                            **_options
                        ),
                        convention.legosc2index(
                            leg_i=leg_pre+1,
                            joint_i=0,
                            **_options
                        ),
                        legs_amplitude_following, phase_follow
                    ])
                    connectivity.append([
                        convention.legosc2index(
                            leg_i=leg_pre+1,
                            joint_i=0,
                            **_options
                        ),
                        convention.legosc2index(
                            leg_i=leg_pre,
                            joint_i=0,
                            **_options
                        ),
                        legs_amplitude_following, -phase_follow
                    ])

        # Body-legs interaction
        for leg_i in range(options.morphology.n_legs//2):
            for side_i in range(2):
                for i in range(n_body_joints):  # [0, 1, 7, 8, 9, 10]
                    for side_leg in range(2): # Muscle facing front/back
                        for lateral in range(2):
                            walk_phase = (
                                0
                                if i in [0, 1, 7, 8, 9, 10]
                                else np.pi
                            )
                            # Forelimbs
                            connectivity.append([
                                convention.bodyosc2index(
                                    joint_i=i,
                                    side=(side_i+lateral)%2
                                ),
                                convention.legosc2index(
                                    leg_i=leg_i,
                                    side_i=side_i,
                                    joint_i=0,
                                    side=(side_i+side_leg)%2
                                ),
                                legs2body_amplitude,
                                (
                                    walk_phase
                                    + np.pi*(side_i+1)
                                    + lateral*np.pi
                                    + side_leg*np.pi
                                    + leg_i*np.pi
                                )
                            ])
        if verbose:
            with np.printoptions(suppress=True, precision=3, threshold=sys.maxsize):
                print("Oscillator connectivity:\n{}".format(np.array(connectivity)))
        return connectivity

    @staticmethod
    def show_connectivity():
        import networkx as nx
        import matplotlib.pyplot as plt
        from .animat_options import AmphibiousOptions
        n_joints = 11
        n_dofs_leg = 4
        n_leg = 4
        dim_body = n_joints * 2
        dim = 2 * n_joints + 2 * n_leg * n_dofs_leg
        options = AmphibiousOptions()
        oscillator_array = np.asarray(
            AmphibiousOscillatorConnectivityArray.from_options(options).array
        )
        contact_array = np.asarray(
            AmphibiousContactsConnectivityArray.from_options(options).array
        )
        graph = nx.DiGraph()
        plt.figure()
        pos = np.zeros([dim, 2])
        scale_factor = 0.5
        offset_leg = 1.5

        for i in np.arange(dim):

            if i < dim_body:
                graph.add_node(i, pos=(-scale_factor, -scale_factor * (i)))
                if i >= n_joints:
                    graph.add_node(i, pos=(scale_factor, -scale_factor * (i - n_joints)))
            if i < dim_body + n_dofs_leg and i >= dim_body:
                graph.add_node(i, pos=(scale_factor * (-i + dim_body) - offset_leg, 0))
            if i < dim_body + 2 * n_dofs_leg and i >= dim_body + n_dofs_leg:
                graph.add_node(i, pos=(scale_factor * (-i + dim_body + n_dofs_leg) - offset_leg, -scale_factor))
            if i < dim_body + 3 * n_dofs_leg and i >= dim_body + 2 * n_dofs_leg:
                graph.add_node(i, pos=(scale_factor * (i - dim_body - 2 * n_dofs_leg) + offset_leg, 0))
            if i < dim_body + 4 * n_dofs_leg and i >= dim_body + 3 * n_dofs_leg:
                graph.add_node(i, pos=(scale_factor * (i - dim_body - 3 * n_dofs_leg) + offset_leg, -scale_factor))
            if i < dim_body + 5 * n_dofs_leg and i >= dim_body + 4 * n_dofs_leg:
                graph.add_node(i, pos=(scale_factor * (-i + dim_body + 4 * n_dofs_leg) - offset_leg, -2))
            if i < dim_body + 6 * n_dofs_leg and i >= dim_body + 5 * n_dofs_leg:
                graph.add_node(i, pos=(scale_factor * (-i + dim_body + 5 * n_dofs_leg) - offset_leg, -2.5))
            if i < dim_body + 7 * n_dofs_leg and i >= dim_body + 6 * n_dofs_leg:
                graph.add_node(i, pos=(scale_factor * (i - dim_body - 6 * n_dofs_leg) + offset_leg, -2))
            if i < dim_body + 8 * n_dofs_leg and i >= dim_body + 7 * n_dofs_leg:
                graph.add_node(i, pos=(scale_factor * (i - dim_body - 7 * n_dofs_leg) + offset_leg, -2.5))

        graph.add_node(dim + 1, pos=(-5, -1), node_color='r')
        graph.add_node(dim + 2, pos=(5, -1))
        graph.add_node(dim + 3, pos=(-5, -2))
        graph.add_node(dim + 4, pos=(5, -2))
        graph.add_weighted_edges_from(oscillator_array[:, 0:3], colors='k')
        graph.add_weighted_edges_from(
             np.vstack((contact_array[:, 0], contact_array[:, 1] + 55, np.zeros(len(contact_array)))).T,
             colors='r')
        graph_pose = nx.get_node_attributes(graph, 'pos')
        M = graph.reverse()
        colors = ['g'] * dim + ['r'] * 4
        nx.draw(M, with_labels=True, node_color=colors, node_size=500, pos=graph_pose)
        plt.axis('equal')
        plt.show()
        return

    @classmethod
    def from_options(cls, options):
        """Parameters for walking"""
        connectivity = cls.set_options(options)
        return cls(np.array(connectivity))

    def update(self, options):
        """Update from options

        :param options: Animat options

        """


class AmphibiousJointsArray(JointsArray):
    """Oscillator array"""

    @staticmethod
    def set_options(options):
        """Walking parameters"""
        j_options = options.control.network.joints
        n_body = options.morphology.n_joints_body
        n_dof_legs = options.morphology.n_dof_legs
        n_legs = options.morphology.n_legs
        n_joints = n_body + n_legs*n_dof_legs
        offsets = np.zeros(n_joints)
        # Body offset
        offsets[:n_body] = options.control.drives.turning
        # Legs walking/swimming
        for leg_i in range(n_legs):
            for i in range(n_dof_legs):
                offsets[n_body + leg_i*n_dof_legs + i] = (
                    j_options.legs_offsets[i].value(
                        options.control.drives
                    )
                )
        # Turning legs
        for leg_i in range(n_legs//2):
            for side in range(2):
                offsets[n_body + 2*leg_i*n_dof_legs + side*n_dof_legs + 0] += (
                    options.control.drives.turning
                    *(1 if leg_i else -1)
                    *(1 if side else -1)
                )
        # Turning body
        for i in range(n_body):
            offsets[i] += j_options.body_offsets[i].value(
                options.control.drives
            )
        rates = 5*np.ones(n_joints)
        return offsets, rates

    @classmethod
    def from_options(cls, options):
        """Parameters for walking"""
        offsets, rates = cls.set_options(options)
        return cls.from_parameters(offsets, rates)

    def update(self, options):
        """Update from options

        :param options: Animat options

        """
        offsets, _ = self.set_options(options)
        self.offsets[:] = offsets


class AmphibiousContactsArray(ContactsArray):
    """Amphibious contacts sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        # n_body = options.morphology.n_joints_body
        n_contacts = options.morphology.n_legs
        # n_joints = options.morphology.n_joints()
        contacts = np.zeros([n_iterations, n_contacts, 9])  # x, y, z
        return cls(contacts)


class AmphibiousContactsConnectivityArray(ConnectivityArray):
    """Amphibious contacts connectivity array"""

    @classmethod
    def from_options(cls, options, verbose=False):
        """Default"""
        connectivity = []
        options_conn = options.control.network.connectivity
        # options.morphology.n_legs
        convention = AmphibiousConvention(options)
        for leg_i in range(options.morphology.n_legs//2):
            for side_i in range(2):
                for joint_i in range(options.morphology.n_dof_legs):
                    for side_o in range(2):
                        for sensor_leg_i in range(options.morphology.n_legs//2):
                            for sensor_side_i in range(2):
                                weight = (
                                    options_conn.weight_sens_contact_e
                                    if (
                                        (leg_i == sensor_leg_i)
                                        != (side_i == sensor_side_i)
                                    )
                                    else options_conn.weight_sens_contact_i
                                )
                                connectivity.append([
                                    convention.legosc2index(
                                        leg_i=leg_i,
                                        side_i=side_i,
                                        joint_i=joint_i,
                                        side=side_o
                                    ),
                                    convention.contactleglink2index(
                                        leg_i=sensor_leg_i,
                                        side_i=sensor_side_i
                                    ),
                                    weight
                                ])
        if verbose:
            print("Contacts connectivity:\n{}".format(np.array(connectivity)))
        if not connectivity:
            connectivity = [[]]
        return cls(np.array(connectivity, dtype=np.float64))


class AmphibiousHydroConnectivityArray(ConnectivityArray):
    """Amphibious hydro connectivity array"""

    @classmethod
    def from_options(cls, options, verbose=False):
        """Default"""
        connectivity = []
        options_conn = options.control.network.connectivity
        # options.morphology.n_legs
        convention = AmphibiousConvention(options)
        for joint_i in range(options.morphology.n_joints_body):
            for side_osc in range(2):
                connectivity.append([
                    convention.bodyosc2index(
                        joint_i=joint_i,
                        side=side_osc
                    ),
                    joint_i+1,
                    options_conn.weight_sens_hydro_freq,
                    options_conn.weight_sens_hydro_amp
                ])
        if verbose:
            print("Hydro connectivity:\n{}".format(np.array(connectivity)))
        return cls(np.array(connectivity, dtype=np.float64))


class AmphibiousProprioceptionArray(ProprioceptionArray):
    """Amphibious proprioception sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        n_joints = options.morphology.n_joints()
        proprioception = np.zeros([n_iterations, n_joints, 9])
        return cls(proprioception)


class AmphibiousGpsArray(GpsArray):
    """Amphibious gps sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        n_links = options.morphology.n_links()
        gps = np.zeros([n_iterations, n_links, 20])
        return cls(gps)


class AmphibiousHydrodynamicsArray(HydrodynamicsArray):
    """Amphibious hydrodynamics sensors array"""

    @classmethod
    def from_options(cls, options, n_iterations):
        """Default"""
        n_body = options.morphology.n_links_body()
        hydrodynamics = np.zeros([n_iterations, n_body, 6])  # Fxyz, Mxyz
        return cls(hydrodynamics)
