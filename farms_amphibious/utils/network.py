"""Network"""

from typing import List
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import (
    colorConverter,
    Normalize,
    ListedColormap,
    LogNorm,
    SymLogNorm,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from farms_core import pylog
from farms_core.analysis.plot import plot_matrix, MatrixLine

from ..data.data import AmphibiousData
from ..data.data_cy import ConnectionType
from ..model.convention import AmphibiousConvention
from ..model.options import AmphibiousOptions  # AmphibiousMorphologyOptions
from ..control.network import NetworkODE


def rotate(vector, theta):
    """Rotate vector"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array(((cos_t, -sin_t), (sin_t, cos_t)))
    return np.dot(rotation, vector)


def direction(vector1, vector2):
    """Unit direction"""
    return (vector2-vector1)/np.linalg.norm(vector2-vector1)


def connect_positions(source, destination, dir_shift, perp_shift):
    """Connect positions"""
    connection_direction = direction(source, destination)
    connection_perp = rotate(connection_direction, 0.5*np.pi)
    new_source = (
        source
        + dir_shift*connection_direction
        + perp_shift*connection_perp
    )
    new_destination = (
        destination
        - dir_shift*connection_direction
        + perp_shift*connection_perp
    )
    return new_source, new_destination


def draw_nodes(positions, radius, color, prefix, show_text=True, **kwargs):
    """Draw nodes"""
    nodes = [
        plt.Circle(
            position,
            radius,
            facecolor=color,
            edgecolor=0.7*np.array(colorConverter.to_rgb(color)),
            linewidth=2,
            animated=True,
            **kwargs,
        )  # fill=False, clip_on=False
        for position in positions
    ]

    nodes_texts = [
        plt.text(
            # position[0]+radius, position[1]+radius,
            position[0], position[1],
            f'{prefix}{i}' if prefix else '',
            # transform=axes.transAxes,
            # va='bottom',
            # ha='left',
            va='center',
            ha='center',
            fontsize=5,
            color='k',
            animated=True,
        ) if show_text else None
        for i, position in enumerate(positions)
    ]

    return nodes, nodes_texts


def draw_connectivity(sources, destinations, connectivity, **kwargs):
    """Draw nodes"""
    radius = kwargs.pop('radius')
    rad = kwargs.pop('rad')
    color = kwargs.pop('color')
    alpha = kwargs.pop('alpha')
    if isinstance(color, ListedColormap):
        color_values = kwargs.pop('color_values')
    for connection_i, connection in enumerate(connectivity):
        assert (
            int(connection[0]+0.5) < len(destinations)
            and int(connection[1]+0.5) < len(sources)
        ), (
            f'Connection connection_i={connection_i}:'
            f'\nconnection[1]={connection[1]} -> connection[0]={connection[0]}'
            f'\nlen(sources)={len(sources)},'
            f' len(destinations)={len(destinations)}'
            f'\nsources={sources}\ndestinations={destinations}'
        )
    node_connectivity = [
        patches.FancyArrowPatch(
            *connect_positions(
                source,
                destination,
                radius,
                -radius/2 if rad > 0 else 0
            ),
            arrowstyle=patches.ArrowStyle(
                stylename='Simple',
                head_length=8,
                head_width=4,
                tail_width=0.5,
            ),
            connectionstyle=patches.ConnectionStyle(
                'Arc3',
                rad=rad,
            ),
            color=colorConverter.to_rgb(
                color(color_values[connection_i])
                if isinstance(color, ListedColormap)
                else color
            )+(alpha,),
        )
        for connection_i, (source, destination) in enumerate([
            [
                sources[int(connection[1]+0.5)],
                destinations[int(connection[0]+0.5)],
            ]
            for connection in connectivity
        ])
    ]
    return node_connectivity


def draw_network(source, destination, radius, connectivity, **kwargs):
    """Draw network"""
    # Arguments
    prefix = kwargs.pop('prefix')
    rad = kwargs.pop('rad')
    color_nodes = kwargs.pop('color_nodes')
    color_arrows = kwargs.pop('color_arrows', None)
    show_text = kwargs.pop('show_text', True)
    options = {}
    alpha = kwargs.pop('alpha')
    weights = kwargs.pop('weights', [])
    if color_arrows is None:
        color_arrows = colorConverter.to_rgb(color_nodes)
    else:
        if list(weights):
            weight_min = np.min(weights)
            weight_max = np.max(weights)
            if weight_max == weight_min:
                weight_max += 1e-3
                weight_min -= 1e-3
            options['color_values'] = (
                (weights-weight_min)/(weight_max-weight_min)
            )
        else:
            options['color_values'] = None

    # Nodes
    nodes, nodes_texts = draw_nodes(
        positions=source,
        radius=radius,
        color=color_nodes,
        prefix=prefix,
        show_text=show_text,
        **kwargs,
    )
    node_connectivity = draw_connectivity(
        sources=source,
        destinations=destination,
        radius=radius,
        connectivity=connectivity,
        rad=rad,
        color=color_arrows,
        alpha=alpha,
        **options,
    )
    return nodes, nodes_texts, node_connectivity


def create_colorbar(cax, cmap, vmin, vmax, **kwargs):
    """Colorbar"""
    return plt.colorbar(
        cax=cax,
        mappable=cm.ScalarMappable(
            norm=Normalize(
                vmin=vmin,
                vmax=vmax,
                clip=True,
            ),
            cmap=cmap,
        ),
        **kwargs,
    )


class NetworkFigure:
    """Network figure"""

    def __init__(self, morphology, data):
        super().__init__()
        self.data = data
        self.morphology = morphology

        # Plot
        self.axes = None
        self.figure = None

        # Artists
        self.oscillators = None
        self.oscillators_texts = None
        self.drives = None
        self.drives_texts = None
        self.joint_sensors = None
        self.joint_sensor_texts = None
        self.contact_sensors = None
        self.contact_sensor_texts = None
        self.xfrc_sensors = None
        self.xfrc_sensor_texts = None

        # Animation
        self.animation = None
        self.timestep = self.data.timestep
        self.time = None
        self.n_iterations = np.shape(self.data.sensors.links.array)[0]
        self.interval = 25
        self.n_frames = round(self.n_iterations*self.timestep / (1e-3*self.interval))
        self.network = NetworkODE(self.data)  # , max_step=self.timestep
        self.cmap_phases = plt.get_cmap('Greens')
        self.cmap_drives = plt.get_cmap('turbo')
        self.cmap_drive_max = 6
        self.cmap_joints = plt.get_cmap('BrBG')
        self.cmap_joint_max = 0.5*np.pi
        self.cmap_contacts = plt.get_cmap('Oranges')
        self.cmap_contact_max = 2e-1
        self.cmap_xfrc = plt.get_cmap('Blues')
        self.cmap_xfrc_max = 2e-2

    def animate(self, convention, **kwargs):
        """Setup animation"""

        # Options
        is_large = convention.n_joints_body < convention.n_legs
        leg_y_offset = kwargs.pop('leg_y_offset', 5 if is_large else 3)
        leg_y_space = kwargs.pop('leg_y_space', 4 if is_large else 1)
        margin_x = kwargs.pop('margin_x', 4-0.2)
        margin_y = kwargs.pop('margin_y', 0.5-0.2)

        plt.figure(num=self.figure.number)
        self.time = plt.text(
            x=0,
            y=-leg_y_offset-convention.n_dof_legs*leg_y_space-margin_y,
            s=f'Time: {0:02.1f} [s]',
            va='bottom',
            ha='left',
            fontsize=16,
            color='k',
            animated=True,
        )

        # Colorbars
        divider = make_axes_locatable(self.axes)
        size = '3%'
        pad = 0.25  # 0.5, 0.05

        # Oscillators
        cax = divider.append_axes('right', size=size, pad=pad)
        cbar = create_colorbar(
            cax=cax,
            cmap=self.cmap_phases,
            vmin=0, vmax=1,
        )
        cbar.ax.set_ylabel('Oscillators output', rotation=270)
        cbar.ax.get_yaxis().labelpad = -20
        cbar.ax.get_yaxis().zorder = 100
        cbar.ax.tick_params(rotation=270)

        # Drives
        cax = divider.append_axes('right', size=size, pad=pad)
        cbar = create_colorbar(
            cax=cax,
            cmap=self.cmap_drives,
            vmin=0, vmax=self.cmap_drive_max,
        )
        cbar.ax.set_ylabel('Drives', rotation=270)
        cbar.ax.get_yaxis().labelpad = -20
        cbar.ax.get_yaxis().zorder = 100
        cbar.ax.tick_params(rotation=270)

        # Joints
        cax = divider.append_axes('right', size=size, pad=pad)
        cbar = create_colorbar(
            cax=cax,
            cmap=self.cmap_joints,
            vmin=-self.cmap_joint_max, vmax=self.cmap_joint_max,
        )
        cbar.ax.set_ylabel('Joints position [rad]', rotation=270)
        cbar.ax.get_yaxis().labelpad = -20
        cbar.ax.get_yaxis().zorder = 100
        cbar.ax.tick_params(rotation=270)

        # Contacts
        cax = divider.append_axes('right', size=size, pad=pad)
        cbar = create_colorbar(
            cax=cax,
            cmap=self.cmap_contacts,
            vmin=0, vmax=self.cmap_contact_max,
        )
        cbar.ax.set_ylabel('Contacts forces [N]', rotation=270)
        cbar.ax.get_yaxis().labelpad = -20
        cbar.ax.get_yaxis().zorder = 100
        cbar.ax.tick_params(rotation=270)

        # Xfrc
        cax = divider.append_axes('right', size=size, pad=pad)
        cbar = create_colorbar(
            cax=cax,
            cmap=self.cmap_xfrc,
            vmin=0, vmax=self.cmap_xfrc_max,
        )
        cbar.set_label('External forces [N]', rotation=270)
        cbar.ax.get_yaxis().labelpad = -20
        cbar.ax.get_yaxis().zorder = 100
        cbar.ax.tick_params(rotation=270)

        # Animation
        self.animation = FuncAnimation(
            fig=self.figure,
            func=self.animation_update,
            frames=np.arange(self.n_frames),
            init_func=self.animation_init,
            blit=False,
            interval=self.interval,
            cache_frame_data=False,
        )

    def animation_elements(self):
        """Animation elements"""
        return [self.time] + (
            self.drives
            + self.drives_texts
            + self.oscillators
            + self.oscillators_texts
            + self.joint_sensors
            + self.joint_sensor_texts
            + self.contact_sensors
            + self.contact_sensor_texts
            + self.xfrc_sensors
            + self.xfrc_sensor_texts
        )

    def animation_init(self):
        """Animation init"""
        return self.animation_elements()

    def animation_update(self, frame):
        """Animation update"""

        # Time
        iteration = np.rint(frame/self.n_frames*(self.n_iterations-1)).astype(int)
        self.time.set_text(f'Time: {frame*1e-3*self.interval:02.1f} [s]')

        # Oscillator
        for oscillator, phase, amplitude in zip(
                self.oscillators,
                self.data.state.phases(iteration),
                self.data.state.amplitudes(iteration),
        ):
            value = (0.5 if amplitude > 1e-3 else amplitude)*(1+np.cos(phase))
            oscillator.set_facecolor(self.cmap_phases(value))

        # Drive
        drives_array = self.data.network.drives.array[iteration]
        for drive_i, drive in enumerate(self.drives):
            value = np.clip(
                a=drives_array[drive_i],
                a_min=0,
                a_max=self.cmap_drive_max,
            )
            drive.set_facecolor(self.cmap_drives(value/self.cmap_drive_max))

        # Joints sensors
        joints = self.data.sensors.joints
        for joint_i, joint in enumerate(self.joint_sensors):
            value = np.clip(
                a=joints.position(iteration, joint_i),
                a_min=-self.cmap_joint_max,
                a_max=+self.cmap_joint_max,
            )
            joint.set_facecolor(self.cmap_joints(value/self.cmap_joint_max))

        # Contacts sensors
        contacts = self.data.sensors.contacts
        assert (
            len(self.contact_sensors) == np.shape(contacts.array)[1]
        ), (
            'Contacts sensors dimensions do not correspond:'
            f'\n{len(self.contact_sensors)}'
            f' != {np.shape(contacts.array)[1]}'
        )
        for sensor_i, contact in enumerate(self.contact_sensors):
            value = np.clip(
                a=np.linalg.norm(contacts.total(iteration, sensor_i)),
                a_min=0,
                a_max=self.cmap_contact_max,
            )
            contact.set_facecolor(self.cmap_contacts(value/self.cmap_contact_max))

        # Xfrcs sensors
        for sensor_i, hydr in enumerate(self.xfrc_sensors):
            value = np.clip(
                np.linalg.norm(
                    self.data.sensors.xfrc.force(iteration, sensor_i)
                ),
                0, self.cmap_xfrc_max,
            )
            hydr.set_facecolor(self.cmap_xfrc(value/self.cmap_xfrc_max))

        return self.animation_elements()

    def plot(self, animat_options, **kwargs):
        """Plot"""
        convention = AmphibiousConvention.from_amphibious_options(animat_options)
        is_large = convention.n_joints_body < convention.n_legs

        # Options
        body_y_offset = kwargs.pop('body_y_offset', 1.5 if is_large else 1)
        body_x_space = kwargs.pop('body_x_space', 4 if is_large else 2)
        leg_x_offset = kwargs.pop('leg_x_offset', 1 if is_large else 3)
        leg_y_offset = kwargs.pop('leg_y_offset', 5 if is_large else 3)
        leg_y_space = kwargs.pop('leg_y_space', 4 if is_large else 1)
        joint_y_space = kwargs.pop('joint_y_space', 2 if is_large else 1)
        contact_y_space = kwargs.pop('contact_y_space', 2 if is_large else 1)
        radius = kwargs.pop('radius', 0.5 if is_large else 0.3)
        margin_x = kwargs.pop('margin_x', 4.5)
        margin_y = kwargs.pop('margin_y', 0.5)
        alpha = kwargs.pop('alpha', 0.3)
        title = kwargs.pop('title', 'Network')
        show_title = kwargs.pop('show_title', True)
        rads = kwargs.pop('rads', [0.02 if is_large else 0.2, 0.0, 0.0])
        use_colorbar = kwargs.pop('use_colorbar', False)
        show_text = kwargs.pop('show_text', True)
        show_drives = kwargs.pop('show_drives', True)
        leg_osc_width = kwargs.pop('leg_osc_width', 1 if is_large else 1)
        figsize = kwargs.pop('figsize', (12, 10))

        # Create figure
        self.figure = plt.figure(num=title, figsize=figsize)
        self.axes = plt.gca()
        self.axes.cla()
        if show_title:
            plt.title(title)
        xlim = (-body_x_space*(convention.n_joints_body-1)-margin_x, margin_x)
        self.axes.set_xlim(xlim)
        self.axes.set_ylim((
            sign*max(
                0.35*(xlim[1] - xlim[0]),
                leg_y_offset+convention.n_dof_legs*leg_y_space+margin_y,
            )
            for sign in [-1, 1]
        ))
        self.axes.set_aspect('equal', adjustable='box')
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        # plt.tight_layout()

        # Colorbar
        if use_colorbar:
            cmap = plt.get_cmap(kwargs.pop('cmap', 'cividis'))

        # Oscillators
        leg_pos = [
            1+(4 if is_large else 2)*split[0]
            for split in np.array_split(
                np.arange(
                    convention.n_joints_body
                    if convention.n_joints_body > 0
                    else convention.n_legs
                ),
                convention.n_legs_pair()+1,
            )[:-1]
        ]
        oscillator_positions = np.array(
            [
                [-body_x_space*osc_x, side_y]
                for osc_x in range(convention.n_joints_body)
                for side_y in [body_y_offset, -body_y_offset]
            ] + [
                [
                    -(leg_x+joint_y+leg_x_offset+osc_side_x),
                    -(leg_y_offset+leg_y_space*joint_y)*side_x,
                ]
                for leg_x in leg_pos
                for side_x in [-1, 1]
                for joint_y in np.arange(convention.n_dof_legs)
                for osc_side_x in (
                        [0]
                        if convention.single_osc_legs
                        else [-leg_osc_width, leg_osc_width]
                )
            ]
        )
        osc_conn_cond = kwargs.pop('osc_conn_cond', lambda osc0, osc1: True)
        connections = (
            np.array([
                [connection[0], connection[1], weight, phase]
                for connection, weight, phase in zip(
                    self.data.network.osc2osc_map.connections.array,
                    self.data.network.osc2osc_map.weights.array,
                    self.data.network.osc2osc_map.desired_phases.array,
                )
                if osc_conn_cond(connection[0], connection[1])
            ])
            if self.data.network.osc2osc_map.connections.array
            else np.empty(0)
        )

        options = {}
        vmin, vmax = 0, 1

        # Oscillator weights
        oscillator_weights = kwargs.pop('oscillator_weights', False)
        use_weights = use_colorbar and oscillator_weights
        if use_weights:
            if connections.any():
                options['weights'] = connections[:, 2]
                vmin = np.min(connections[:, 2])
                vmax = np.max(connections[:, 2])

        # Oscillator phases
        oscillator_phases = kwargs.pop('oscillator_phases', False)
        use_weights = use_colorbar and oscillator_phases
        if use_weights:
            if connections.any():
                options['phases'] = connections[:, 3]
                vmin = np.min(connections[:, 3])
                vmax = np.max(connections[:, 3])

        self.oscillators, self.oscillators_texts, oscillators_connectivity = draw_network(
            source=oscillator_positions,
            destination=oscillator_positions,
            radius=radius,
            connectivity=connections,
            prefix='O',
            rad=rads[0],
            color_nodes='C2',
            color_arrows=cmap if use_weights else None,
            alpha=alpha,
            show_text=show_text,
            **options,
        )

        # Drives
        drives_front = len(animat_options.control.network.drives) == 2
        drives_positions = [] if not show_drives else np.array([
            [2, +2],
            [2, -2],
        ]) if drives_front else oscillator_positions
        self.drives, self.drives_texts, drive2osc_map = draw_network(
            source=drives_positions,
            destination=oscillator_positions,
            radius=1.2*radius if drives_front else 1.5*radius,
            connectivity=[],
            prefix='D' if drives_front else '',
            rad=rads[1],
            color_nodes='C0',
            color_arrows=None,
            alpha=alpha,
            show_text=show_text,
            weights=[],
            zorder=0,
        )

        # Joints data
        joints_positions = np.array(
            [
                [-body_x_space*osc_x, 0]
                for osc_x in range(convention.n_joints_body)
            ] + [
                [
                    -(leg_x+joint_y+leg_x_offset),
                    -(leg_y_offset+leg_y_space*joint_y)*side_x,
                ]
                for leg_x in leg_pos
                for side_x in [-1, 1]
                for joint_y in np.arange(convention.n_dof_legs)
            ]
        )
        joint_conn_cond = kwargs.pop(
            'joint_conn_cond',
            lambda osc0, osc1: True
        )
        connections = np.array([
            [connection[0], connection[1], weight]
            for connection, weight in zip(
                self.data.network.joints2osc_map.connections.array,
                self.data.network.joints2osc_map.weights.array,
            )
            if joint_conn_cond(connection[0], connection[1])
        ]) if self.data.network.joints2osc_map.connections.array else np.empty(0)

        # Joints plot
        options = {}
        joints_weights = kwargs.pop('joints_weights', False)
        use_weights = use_colorbar and joints_weights
        if use_weights:
            if connections.any():
                options['weights'] = connections[:, 2]
                vmin = np.min(connections[:, 2])
                vmax = np.max(connections[:, 2])
            else:
                options['weights'] = []
                vmin, vmax = 0, 1
        self.joint_sensors, self.joint_sensor_texts, joint2osc_map = draw_network(
            source=joints_positions,
            destination=oscillator_positions,
            radius=0.7*radius,
            connectivity=connections,
            prefix='J',
            rad=rads[1],
            color_nodes='C7',
            color_arrows=cmap if use_weights else None,
            alpha=alpha,
            show_text=show_text,
            **options,
        )

        # Contacts
        contacts_positions = np.array(
            [
                [
                    -(leg_x+convention.n_dof_legs-1+leg_x_offset),
                    side_y*(leg_y_offset+(
                        convention.n_dof_legs-1
                    )*leg_y_space+contact_y_space),
                ]
                for leg_x in leg_pos
                for side_y in [1, -1]
            ] + [
                [-(body_x_space*(osc_x+0.5)-0.3), 0]
                for osc_x in range(-1, convention.n_joints_body)
            ]
        ) if animat_options.control.sensors.contacts else []
        contact_conn_cond = kwargs.pop(
            'contact_conn_cond',
            lambda osc0, osc1: True
        )
        connections = np.array([
            [connection[0], connection[1], weight]
            for connection, weight in zip(
                self.data.network.contacts2osc_map.connections.array,
                self.data.network.contacts2osc_map.weights.array,
            )
            if contact_conn_cond(connection[0], connection[1])
        ]) if self.data.network.contacts2osc_map.connections.array else np.empty(0)
        options = {}
        contacts_weights = kwargs.pop('contacts_weights', False)
        use_weights = use_colorbar and contacts_weights
        if use_weights:
            if connections.any():
                options['weights'] = connections[:, 2]
                vmin = np.min(connections[:, 2])
                vmax = np.max(connections[:, 2])
            else:
                options['weights'] = []
                vmin, vmax = 0, 1
        self.contact_sensors, self.contact_sensor_texts, contact2osc_map = draw_network(
            source=contacts_positions,
            destination=oscillator_positions,
            radius=0.7*radius,
            connectivity=connections,
            prefix='C',
            rad=rads[1],
            color_nodes='C1',
            color_arrows=cmap if use_weights else None,
            alpha=alpha,
            show_text=show_text,
            **options,
        )

        # Xfrc
        xfrc_positions = np.array([
            [-(body_x_space*(osc_x+0.5)+0.3), 0]
            for osc_x in range(-1, convention.n_joints_body)
        ])
        xfrc_conn_cond = kwargs.pop('xfrc_conn_cond', lambda osc0, osc1: True)
        connections = np.array([
            [connection[0], connection[1], connection[2], weight]
            for connection, weight in zip(
                self.data.network.xfrc2osc_map.connections.array,
                self.data.network.xfrc2osc_map.weights.array,
            )
            if xfrc_conn_cond(connection[0], connection[1])
        ]) if self.data.network.xfrc2osc_map.connections.array else np.empty(0)
        options = {}
        xfrc_frequency_weights = kwargs.pop('xfrc_frequency_weights', False)
        xfrc_amplitude_weights = kwargs.pop('xfrc_amplitude_weights', False)
        use_weights = (
            use_colorbar
            and (xfrc_frequency_weights or xfrc_amplitude_weights)
        )
        if use_weights:
            if connections.any():
                options['weights'] = [
                    connection[3]
                    for connection in connections
                    if (
                        xfrc_frequency_weights
                        and connection[2] in (
                            ConnectionType.LATERAL2FREQTEGOTAE,
                            ConnectionType.LATERAL2FREQCOS,
                            ConnectionType.LATERAL2FREQ,
                        )
                    ) or (
                        xfrc_amplitude_weights
                        and connection[2] == ConnectionType.LATERAL2AMP
                    )
                ]
                if options['weights']:
                    vmin = np.min(options['weights'])
                    vmax = np.max(options['weights'])
                else:
                    vmin, vmax = 0, 1
            else:
                options['weights'] = []
                vmin, vmax = 0, 1
        self.xfrc_sensors, self.xfrc_sensor_texts, xfrc2osc_map = draw_network(
            source=xfrc_positions,
            destination=oscillator_positions,
            radius=0.7*radius,
            connectivity=[
                connection
                for connection in self.data.network.xfrc2osc_map.connections.array
                # if xfrc_conn_cond(connection[0], connection[1])
                if (not xfrc_frequency_weights and not xfrc_amplitude_weights)
                or (
                    xfrc_frequency_weights
                    and connection[2] in (
                        ConnectionType.LATERAL2FREQTEGOTAE,
                        ConnectionType.LATERAL2FREQCOS,
                        ConnectionType.LATERAL2FREQ,
                    )
                ) or (
                    xfrc_amplitude_weights
                    and connection[2] == ConnectionType.LATERAL2AMP
                )
            ],
            prefix='H',
            rad=rads[2],
            color_nodes='C3',
            color_arrows=cmap if use_weights else None,
            alpha=2*alpha,
            show_text=show_text,
            **options,
        ) if self.data.network.xfrc2osc_map.connections.array else [[]]*3

        # Show elements
        show_elements = [
            show_oscillators,
            show_joints,
            show_contacts,
            show_xfrc,
            show_oscillators_connectivity,
            show_drives_connectivity,
            show_joints_connectivity,
            show_contacts_connectivity,
            show_xfrc_connectivity,
        ] = [
            kwargs.pop(key, True)
            for key in [
                'show_oscillators',
                'show_joints',
                'show_contacts',
                'show_xfrc',
                'show_oscillators_connectivity',
                'show_drives_connectivity',
                'show_joints_connectivity',
                'show_contacts_connectivity',
                'show_xfrc_connectivity',
            ]
        ]
        assert not kwargs, kwargs
        if show_oscillators_connectivity:
            for arrow in oscillators_connectivity:
                self.axes.add_artist(arrow)
        if show_drives_connectivity:
            for arrow in drive2osc_map:
                self.axes.add_artist(arrow)
        if show_joints_connectivity:
            for arrow in joint2osc_map:
                self.axes.add_artist(arrow)
        if show_contacts_connectivity:
            for arrow in contact2osc_map:
                self.axes.add_artist(arrow)
        if show_xfrc_connectivity:
            for arrow in xfrc2osc_map:
                self.axes.add_artist(arrow)
        if show_oscillators:
            for circle, text in zip(
                    self.oscillators,
                    self.oscillators_texts,
            ):
                self.axes.add_artist(circle)
                if show_text:
                    self.axes.add_artist(text)
        if show_drives:
            for circle, text in zip(
                    self.drives,
                    self.drives_texts,
            ):
                self.axes.add_artist(circle)
                if show_text:
                    self.axes.add_artist(text)
        if show_joints:
            for circle, text in zip(
                    self.joint_sensors,
                    self.joint_sensor_texts,
            ):
                self.axes.add_artist(circle)
                if show_text:
                    self.axes.add_artist(text)
        if show_contacts:
            for circle, text in zip(
                    self.contact_sensors,
                    self.contact_sensor_texts,
            ):
                self.axes.add_artist(circle)
                if show_text:
                    self.axes.add_artist(text)
        if show_xfrc:
            for circle, text in zip(
                    self.xfrc_sensors,
                    self.xfrc_sensor_texts,
            ):
                self.axes.add_artist(circle)
                if show_text:
                    self.axes.add_artist(text)
        if any(show_elements) and use_colorbar:
            pylog.debug('Setting colormap for %s: %s, %s', title, vmin, vmax)
            divider = make_axes_locatable(self.axes)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            create_colorbar(cax=cax, cmap=cmap, vmin=vmin, vmax=vmax)

        return self.figure


def conn_body2body(info, osc0, osc1):
    """Body to body"""
    return info(osc0)['body'] and info(osc1)['body']


def conn_leg2body(info, osc0, osc1):
    """Leg to body"""
    return info(osc0)['body'] and not info(osc1)['body']


def conn_body2leg(info, osc0, osc1):
    """Body to leg"""
    return not info(osc0)['body'] and info(osc1)['body']


def conn_leg2leg(info, osc0, osc1):
    """Leg to leg"""
    return not info(osc0)['body'] and not info(osc1)['body']


def plot_networks_maps(
        data: AmphibiousData,
        animat_options: AmphibiousOptions,
        show_all: bool = False,
        animation_only: bool = False,
        show_text: bool = True,
        **kwargs,
):
    """Plot network maps"""

    # Plots
    plots = {}

    # Plot network
    morphology = animat_options.morphology
    convention = AmphibiousConvention.from_amphibious_options(animat_options)
    network_anim = NetworkFigure(morphology, data)
    plots['network_complete'] = network_anim.plot(
        title='Complete network',
        animat_options=animat_options,
        show_title=False,
        # use_colorbar=True,
        # oscillator_weights=False,
        show_text=show_text,
        **kwargs,
    )
    network_anim.animate(convention)

    if animation_only:
        return network_anim, plots

    network = NetworkFigure(morphology, data)
    plots['network_oscillators_standalone'] = network.plot(
        title='Network connectivity',
        show_title=False,
        animat_options=animat_options,
        show_contacts_connectivity=False,
        show_xfrc_connectivity=False,
        use_colorbar=False,
        oscillator_weights=False,
        **kwargs,
    )

    network = NetworkFigure(morphology, data)
    plots['network_oscillators_standalone_nodrives'] = network.plot(
        title='Network connectivity (no_drives)',
        show_title=False,
        animat_options=animat_options,
        show_contacts_connectivity=False,
        show_xfrc_connectivity=False,
        use_colorbar=False,
        oscillator_weights=False,
        show_drives=False,
        **kwargs,
    )

    if show_all:

        info = convention.oscindex2information
        body2body = partial(conn_body2body, info)
        leg2body = partial(conn_leg2body, info)
        body2leg = partial(conn_body2leg, info)
        leg2leg = partial(conn_leg2leg, info)

        leg2sameleg = (
            lambda osc0, osc1: (
                not info(osc0)['body']
                and not info(osc1)['body']
                and info(osc0)['leg'] == info(osc1)['leg']
            )
        )
        leg2diffleg = (
            lambda osc0, osc1: (
                not info(osc0)['body']
                and not info(osc1)['body']
                and info(osc0)['leg'] != info(osc1)['leg']
            )
        )
        contact2sameleg = (
            lambda osc0, osc1: (
                not info(osc0)['body']
                and info(osc0)['leg'] == osc1
                # and osc1 in (1, 2)
            )
        )
        contact2diffleg = (
            lambda osc0, osc1: (
                not info(osc0)['body']
                and info(osc0)['leg'] != osc1
                # and osc1 in (1, 2)
            )
        )

        # Plot network oscillator weights connectivity
        plots['network_oscillators'] = network.plot(
            title='Oscillators complete connectivity',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            use_colorbar=True,
            oscillator_weights=True,
            **kwargs,
        )
        plots['network_oscillators_body2body'] = network.plot(
            title='Oscillators body2body connectivity',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=body2body,
            use_colorbar=True,
            oscillator_weights=True,
            **kwargs,
        )
        plots['network_oscillators_body2limb'] = network.plot(
            title='Oscillators body2limb connectivity',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=body2leg,
            use_colorbar=True,
            oscillator_weights=True,
            **kwargs,
        )
        plots['network_oscillators_limb2body'] = network.plot(
            title='Oscillators limb2body connectivity',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=leg2body,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_weights=True,
            **kwargs,
        )
        plots['network_oscillators_limb2limb'] = network.plot(
            title='Oscillators limb2limb connectivity',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=leg2leg,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_weights=True,
            **kwargs,
        )
        plots['network_oscillators_intralimb'] = network.plot(
            title='Oscillators intralimb connectivity',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=leg2sameleg,
            rads=[0.2, 0.0, 0.0],
            use_colorbar=True,
            oscillator_weights=True,
            **kwargs,
        )
        plots['network_oscillators_interlimb'] = network.plot(
            title='Oscillators interlimb connectivity',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=leg2diffleg,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_weights=True,
        )

        # Plot network oscillator phases connectivity
        plots['network_phases'] = network.plot(
            title='Oscillators complete phases',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            use_colorbar=True,
            oscillator_phases=True,
            **kwargs,
        )
        plots['network_phases_body2body'] = network.plot(
            title='Oscillators body2body phases',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=body2body,
            use_colorbar=True,
            oscillator_phases=True,
            **kwargs,
        )
        plots['network_phases_body2limb'] = network.plot(
            title='Oscillators body2limb phases',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=body2leg,
            use_colorbar=True,
            oscillator_phases=True,
            **kwargs,
        )
        plots['network_phases_limb2body'] = network.plot(
            title='Oscillators limb2body phases',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=leg2body,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_phases=True,
            **kwargs,
        )
        plots['network_phases_limb2limb'] = network.plot(
            title='Oscillators limb2limb phases',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=leg2leg,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_phases=True,
            **kwargs,
        )
        plots['network_phases_intralimb'] = network.plot(
            title='Oscillators intralimb phases',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=leg2sameleg,
            rads=[0.2, 0.0, 0.0],
            use_colorbar=True,
            oscillator_phases=True,
            **kwargs,
        )
        plots['network_phases_interlimb'] = network.plot(
            title='Oscillators interlimb phases',
            animat_options=animat_options,
            show_contacts_connectivity=False,
            show_xfrc_connectivity=False,
            osc_conn_cond=leg2diffleg,
            rads=[0.05, 0.0, 0.0],
            use_colorbar=True,
            oscillator_phases=True,
            **kwargs,
        )

        # Plot contacts connectivity
        plots['network_contacts'] = network.plot(
            title='Contacts complete connectivity',
            animat_options=animat_options,
            show_oscillators_connectivity=False,
            show_xfrc_connectivity=False,
            use_colorbar=True,
            contacts_weights=True,
            **kwargs,
        )
        plots['network_contacts_intralimb'] = network.plot(
            title='Contacts intralimb connectivity',
            animat_options=animat_options,
            show_oscillators_connectivity=False,
            show_xfrc_connectivity=False,
            contact_conn_cond=contact2sameleg,
            use_colorbar=True,
            contacts_weights=True,
            **kwargs,
        )
        plots['network_contacts_interlimb'] = network.plot(
            title='Contacts interlimb connectivity',
            animat_options=animat_options,
            show_oscillators_connectivity=False,
            show_xfrc_connectivity=False,
            contact_conn_cond=contact2diffleg,
            use_colorbar=True,
            contacts_weights=True,
            **kwargs,
        )

        # Plot xfrc connectivity
        plots['network_xfrc'] = network.plot(
            title='Xfrc complete connectivity',
            animat_options=animat_options,
            show_oscillators_connectivity=False,
            show_contacts_connectivity=False,
            **kwargs,
        )
        plots['network_xfrc_frequency'] = network.plot(
            title='Xfrc frequency connectivity',
            animat_options=animat_options,
            show_oscillators_connectivity=False,
            show_contacts_connectivity=False,
            use_colorbar=True,
            xfrc_frequency_weights=True,
            **kwargs,
        )
        plots['network_xfrc_amplitude'] = network.plot(
            title='Xfrc amplitude connectivity',
            animat_options=animat_options,
            show_oscillators_connectivity=False,
            show_contacts_connectivity=False,
            use_colorbar=True,
            xfrc_amplitude_weights=True,
            **kwargs,
        )

    return network_anim, plots


def get_osc_matrices(
        animat_options: AmphibiousOptions,
        osc_names: List[str],
):
    """Get oscillator matrices"""

    n_osc = len(osc_names)
    osc_weights = np.full((n_osc, n_osc), np.nan)
    osc_phase_bias = np.full((n_osc, n_osc), np.nan)

    # Osc2osc connections
    for connection in animat_options.control.network.osc2osc:
        assert connection['type'] == 'OSC2OSC'
        osc_weights[
            osc_names.index(connection['in']),
            osc_names.index(connection['out']),
        ] = connection['weight']
        osc_phase_bias[
            osc_names.index(connection['in']),
            osc_names.index(connection['out']),
        ] = connection['phase_bias']

    return osc_weights, osc_phase_bias


def plot_connectivity_osc_couplings(
        animat_options: AmphibiousOptions,
        osc_names: List[str],
        osc_labels: List[str],
        osc_lines: List[MatrixLine],
        osc_twin: List,
):
    """Plot oscillator couplings"""

    plots = {}

    osc_weights, osc_phase_bias = get_osc_matrices(animat_options, osc_names)

    # Plotting function
    partial_osc_matrix = partial(
        plot_matrix,
        labels=[osc_labels, osc_labels],
        xlabel='Emitters',
        ylabel='Receiving oscillators',
        lines=osc_lines,
        xtwin=osc_twin,
        ytwin=osc_twin,
        interpolation='nearest',
        aspect='auto',
    )

    # Oscillator coupling weights
    name = 'network_matrix_osc_weights'
    plots[name] = partial_osc_matrix(
        osc_weights,
        fig_name=name,
        clabel='Weight',
        cmap=cm.get_cmap('cividis'),
        norm=LogNorm(),
    )

    # Oscillator coupling phase bias
    name = 'network_matrix_osc_phase_bias'
    plots[name] = partial_osc_matrix(
        (osc_phase_bias + np.pi) % (2*np.pi) - np.pi,
        fig_name=name,
        clabel='Phase bias [rad]',
        cmap=cm.get_cmap('turbo'),  # BrBG
        norm=Normalize(
            vmin=-np.pi,
            vmax=+np.pi,
            clip=True,
        ),
    )

    return plots


def get_sensor_names(
        connections: List,
        designator: str,
):
    """Get sensor names"""
    sensors_names = list(dict.fromkeys([
        connection['out']
        for connection in connections
    ]))
    sensor_labels = [
        designator + (
            sensor[0]
            if isinstance(sensor, tuple)
            else sensor
        ).replace(
            'sensor_', '',
        ).replace(
            'body', 'b',
        ).replace(
            'leg', 'l',
        ).replace(
            '_', ',',
        ) + r'}$'
        for sensor in sensors_names
    ]
    return sensors_names, sensor_labels


def get_sensor_matrix(
        connections: List,
        connection_type: str,
        osc_names: List[str],
        sensors_names: List[str],
):
    """Get sensors matrix"""
    n_osc = len(osc_names)
    n_sensors = len(sensors_names)
    matrix = np.full((n_osc, n_sensors), np.nan)
    for connection in connections:
        if connection['type'] == connection_type:
            matrix[
                osc_names.index(connection['in']),
                sensors_names.index(connection['out']),
            ] = connection['weight']
    return matrix


def plot_connectivity_joints(
        animat_options: AmphibiousOptions,
        osc_names: List[str],
        osc_labels: List[str],
        osc_lines: List[MatrixLine],
        osc_twin: List,
):
    """Plot joints connectivity"""

    plots = {}
    joints_names, joint_labels = get_sensor_names(
        connections=animat_options.control.network.joint2osc,
        designator=r'$\mathcal{J}_{',
    )

    # Joint2osc connections
    partial_joint_matrix = partial(
        plot_matrix,
        labels=[osc_labels, joint_labels],
        xlabel='Emitters',
        ylabel='Receiving oscillators',
        lines=osc_lines,
        ytwin=osc_twin,
        clabel='Weight',
        interpolation='nearest',
        aspect='auto',
        cmap=cm.get_cmap('cividis'),
    )
    for connection_type in [
            'POS2FREQ',
            'VEL2FREQ',
            'TOR2FREQ',
            'POS2AMP',
            'VEL2AMP',
            'TOR2AMP',
            'STRETCH2FREQPOS',
            'STRETCH2FREQNEG',
            'STRETCH2AMPPOS',
            'STRETCH2AMPNEG',
            'STRETCH2FREQTEGOTAE',
            'STRETCH2AMPTEGOTAE',
            'STRETCH2FREQTEGOTAEMOD',
            'STRETCH2AMPTEGOTAEMOD',
            'STRETCH2FREQ',
            'STRETCH2AMP',
            'STRETCHVEL2FREQ',
            'STRETCHVEL2AMP',
    ]:
        matrix = get_sensor_matrix(
            connections=animat_options.control.network.joint2osc,
            connection_type=connection_type,
            osc_names=osc_names,
            sensors_names=joints_names,
        )
        name = f'network_matrix_j2o_{connection_type}'
        plots[name] = partial_joint_matrix(matrix, fig_name=name)
        name += '_reducex'
        plots[name] = partial_joint_matrix(matrix, fig_name=name, reduce_x=True)

    return plots


def plot_connectivity_contacts(
        animat_options: AmphibiousOptions,
        osc_names: List[str],
        osc_labels: List[str],
        osc_lines: List[MatrixLine],
        osc_twin: List,
):
    """Plot contacts connectivity"""

    plots = {}
    contacts_names, contact_labels = get_sensor_names(
        connections=animat_options.control.network.contact2osc,
        designator=r'$\mathcal{C}_{',
    )

    # Contact2osc connections
    partial_contact_matrix = partial(
        plot_matrix,
        labels=[osc_labels, contact_labels],
        xlabel='Emitters',
        ylabel='Receiving oscillators',
        lines=osc_lines,
        ytwin=osc_twin,
        clabel='Weight',
        interpolation='nearest',
        aspect='auto',
        cmap=cm.get_cmap('cividis'),
    )
    for connection_type in [
            'REACTION2FREQ',
            'REACTION2AMP',
            'FRICTION2FREQ',
            'REACTION2PHASE0',
            'REACTION2PHASEPI',
            'REACTION2FREQTEGOTAE',
            'FRICTION2AMP',
    ]:
        matrix = get_sensor_matrix(
            connections=animat_options.control.network.contact2osc,
            connection_type=connection_type,
            osc_names=osc_names,
            sensors_names=contacts_names,
        )
        name = f'network_matrix_c2o_{connection_type}'
        plots[name] = partial_contact_matrix(matrix, fig_name=name)
        name += '_reducex'
        plots[name] = partial_contact_matrix(matrix, fig_name=name, reduce_x=True)

    return plots


def plot_connectivity_xfrc(
        animat_options: AmphibiousOptions,
        osc_names: List[str],
        osc_labels: List[str],
        osc_lines: List[MatrixLine],
        osc_twin: List,
):
    """Plot xfrc connectivity"""

    plots = {}
    xfrc_names, xfrc_labels = get_sensor_names(
        connections=animat_options.control.network.xfrc2osc,
        designator=r'$\mathcal{F}_{',
    )

    # Xfrc2osc connections
    partial_xfrc_matrix = partial(
        plot_matrix,
        labels=[osc_labels, xfrc_labels],
        xlabel='Emitters',
        ylabel='Receiving oscillators',
        lines=osc_lines,
        ytwin=osc_twin,
        clabel='Weight',
        interpolation='nearest',
        aspect='auto',
        cmap=cm.get_cmap('cividis'),
    )
    for connection_type in [
            'LATERAL2FREQTEGOTAE',
            'LATERAL2FREQCOS',
            'LATERAL2FREQ',
            'LATERAL2AMP',
    ]:
        matrix = get_sensor_matrix(
            connections=animat_options.control.network.xfrc2osc,
            connection_type=connection_type,
            osc_names=osc_names,
            sensors_names=xfrc_names,
        )
        name = f'network_matrix_j2o_{connection_type}'
        plots[name] = partial_xfrc_matrix(matrix, fig_name=name)
        name += '_reducex'
        plots[name] = partial_xfrc_matrix(matrix, fig_name=name, reduce_x=True)

    return plots


def plot_connectivity_sensors(**kwargs):
    """Plot sensors connectivity"""
    plots = {}
    plots.update(plot_connectivity_joints(**kwargs))
    plots.update(plot_connectivity_contacts(**kwargs))
    plots.update(plot_connectivity_xfrc(**kwargs))
    return plots


def plot_connectivity_drive(
        animat_options: AmphibiousOptions,
        osc_names: List[str],
        osc_labels: List[str],
        osc_lines: List[MatrixLine],
        osc_twin: List,
):
    """Plot drives connectivity"""

    plots = {}
    contacts_names = animat_options.control.sensors.contacts
    contact_labels = [
        r'$\mathcal{C}_{' + (
            sensor[0]
            if isinstance(sensor, tuple)
            else sensor
        ).replace(
            'sensor_', '',
        ).replace(
            'body', 'b',
        ).replace(
            'leg', 'l',
        ).replace(
            '_', ',',
        ) + r'}$'
        for sensor in contacts_names
    ]

    # Matrix
    n_osc = len(osc_names)
    n_contacts = len(contacts_names)
    n_drives = len(animat_options.control.network.drives)
    if n_drives == n_osc:
        n_drives = 1
    matrix = np.full((n_osc, n_drives+n_contacts), np.nan)
    for osc_i in range(n_osc):
        drive_i = animat_options.control.network.drive2osc[osc_i]
        drive = animat_options.control.network.drives[drive_i]
        if n_drives == 1:
            assert osc_i == drive_i, f'{osc_i=} != {drive_i=}'
            matrix[osc_i, 0] = 1
        else:
            matrix[osc_i, drive_i] = 1
        for contact in drive.contacts:
            assert contact in contacts_names, (
                f'{contact=} not in {contacts_names=}'
            )
            matrix[osc_i, n_drives + contacts_names.index(contact)] = 1

    # Contact2osc connections
    partial_contact_matrix = partial(
        plot_matrix,
        labels=[
            osc_labels,
            (
                [r'$\mathcal{D}_{intrinsic}$']
                if n_drives == 1
                else osc_labels
            ) + contact_labels
        ],
        xlabel='Emitters',
        ylabel='Receiving oscillators',
        clabel='Weight',
        interpolation='nearest',
        aspect='auto',
        cmap=cm.get_cmap('cividis'),
    )

    name = 'network_matrix_d2o'
    plots[name] = partial_contact_matrix(matrix, fig_name=name)
    name += '_reducex'
    plots[name] = partial_contact_matrix(matrix, fig_name=name, reduce_x=True)

    return plots


def plot_connectivity_full(
        animat_options: AmphibiousOptions,
        osc_names: List[str],
        osc_labels: List[str],
        osc_lines: List[MatrixLine],
        osc_twin: List,
):
    """Plot full connectivity"""

    plots = {}
    joints_weights, joints_names, joint_labels = {}, {}, {}
    contacts_weights, contacts_names, contact_labels = {}, {}, {}

    # Oscillators
    osc_weights, _osc_phase_bias = get_osc_matrices(animat_options, osc_names)

    # Joints
    seps = []
    desinators = []
    for conn_type, des in [
            ('STRETCH2FREQTEGOTAE', r'\theta'),
            ('STRETCHVEL2FREQ', r'\dot{\theta}'),
    ]:
        joints_names[conn_type], joint_labels[conn_type] = get_sensor_names(
            connections=animat_options.control.network.joint2osc,
            designator=r'$\mathcal{J}_{' + des + ',',
        )
        joints_weights[conn_type] = get_sensor_matrix(
            connections=animat_options.control.network.joint2osc,
            connection_type=conn_type,
            osc_names=osc_names,
            sensors_names=joints_names[conn_type],
        )
        seps += [joints_weights[conn_type].shape[1]]
        desinators += [r'$\mathcal{J}_{' + des + r'}$']

    # Contacts
    for conn_type, des in [
            ('REACTION2PHASE0', 'E'),
            ('REACTION2PHASEPI', 'I'),
    ]:
        contacts_names[conn_type], contact_labels[conn_type] = get_sensor_names(
            connections=animat_options.control.network.contact2osc,
            designator=r'$\mathcal{C}_{' + des + ',',
        )
        contacts_weights[conn_type] = get_sensor_matrix(
            connections=animat_options.control.network.contact2osc,
            connection_type=conn_type,
            osc_names=osc_names,
            sensors_names=contacts_names[conn_type],
        )
        seps += [contacts_weights[conn_type].shape[1]]
        desinators += [r'$\mathcal{C}_{' + des + r'}$']

    # Plotting function
    sensors_labels = [
        label
        for labels in joint_labels.values()
        for label in labels
    ]+[
        label
        for labels in contact_labels.values()
        for label in labels
    ]
    matrix = np.concatenate(
        (
            (osc_weights,)
            + tuple(joints_weights.values())
            + tuple(contacts_weights.values())
        ),
        axis=1,
    )
    matrix_pos = np.where(matrix > 0, matrix, np.nan)
    matrix_neg = np.where(matrix < 0, matrix, np.nan)
    if not np.isnan(matrix_neg).all() and not np.isnan(matrix_pos).all():
        vmax = np.nanmax(matrix_pos)
        vmin = np.nanmin(matrix_neg)
        linthresh = min(-np.nanmax(matrix_neg), np.nanmin(matrix_pos))
        norm = SymLogNorm(
            linthresh=linthresh,
            linscale=0.5,
            vmin=vmin,
            vmax=vmax,
            base=10,
        )
    else:
        norm = LogNorm()
    partial_full_matrix = partial(
        plot_matrix,
        matrix=matrix,
        labels=[osc_labels, osc_labels+sensors_labels],
        xlabel='Emitters',
        ylabel='Receiving oscillators',
        lines=osc_lines+[
            # Separation between oscillator coupling and sensory feedback
            MatrixLine.column(
                index=len(osc_labels)-0.5,
                color=(1, 0, 0, 0.5),
                linestyle='dashed',
                linewidth=1,
            )
        ] + [
            MatrixLine.column(
                index=len(osc_labels)-0.5+sum(seps[:sep_i])+sep,
                color=(0, 0, 0, 0.3),
                linestyle='dashed',
                linewidth=1,
            )
            for sep_i, sep in enumerate(seps[:-1])
        ],
        xtwin=osc_twin + [
            [
                designator,
                [
                    len(osc_labels)+sum(seps[:des_i]),
                    len(osc_labels)+sum(seps[:des_i+1]),
                ]
            ]
            for des_i, designator in enumerate(desinators)
        ],
        ytwin=osc_twin,
        interpolation='nearest',
        aspect='auto',
        clabel='Weight',
        cmap=cm.get_cmap('turbo'),
        norm=norm,
    )

    # Full weights connectivity
    name = 'network_matrix_full'
    plots[name] = partial_full_matrix(fig_name=name)
    plots[name+'_reducex'] = partial_full_matrix(
        fig_name=name+'_reducex',
        reduce_x=True,
    )
    plots[name+'_reducey'] = partial_full_matrix(
        fig_name=name+'_reducey',
        reduce_y=True,
    )

    return plots


def plot_connectivity_matrix(
        # data: AmphibiousData,
        animat_options: AmphibiousOptions,
):
    """Plot connectivity matrix"""

    plots = {}
    convention = AmphibiousConvention.from_amphibious_options(animat_options)

    # Oscillators
    n_osc = convention.n_osc()
    n_osc_leg = convention.n_osc_leg()
    n_osc_body = convention.n_osc_body()
    osc_names = [convention.oscindex2name(index) for index in range(n_osc)]
    osc_labels = [
        '$' + name.replace(
            'body', 'b',
        ).replace(
            'leg', 'l',
        ).replace(
            '_', ',',
        ).replace(
            'osc,', r'\mathcal{O}_{',
        ) + '}$'
        for name in osc_names
    ]
    osc_twin=[
        ['Body', [0, n_osc_body-1]],
        ['LF', [n_osc_body+0*n_osc_leg, n_osc_body+1*n_osc_leg-1]],
        ['RF', [n_osc_body+1*n_osc_leg, n_osc_body+2*n_osc_leg-1]],
        ['LH', [n_osc_body+2*n_osc_leg, n_osc_body+3*n_osc_leg-1]],
        ['RH', [n_osc_body+3*n_osc_leg, n_osc_body+4*n_osc_leg-1]],
    ]

    # Lines
    osc_lines_row, osc_lines_column = [
        [  # Body
            function(
                index=n_osc_body-0.5,
                color=(1, 0, 0, 0.5),
                linestyle='dashed',
                linewidth=1,
            )
        ] + [  # Limbs
            function(
                index=n_osc_body+(1+i)*n_osc_leg-0.5,
                color=(0, 0, 0, 0.3),
                linestyle='dashed',
                linewidth=1,
            )
            for i in range(max(0, convention.n_legs-1))
        ]
        for function in (MatrixLine.row, MatrixLine.column)
    ]

    # Oscillators
    plots.update(plot_connectivity_osc_couplings(
        animat_options=animat_options,
        osc_names=osc_names,
        osc_labels=osc_labels,
        osc_lines=osc_lines_row+osc_lines_column,
        osc_twin=osc_twin,
    ))

    # Sensors
    plots.update(plot_connectivity_sensors(
        animat_options=animat_options,
        osc_names=osc_names,
        osc_labels=osc_labels,
        osc_lines=osc_lines_row,
        osc_twin=osc_twin,
    ))

    # Drive
    plots.update(plot_connectivity_drive(
        animat_options=animat_options,
        osc_names=osc_names,
        osc_labels=osc_labels,
        osc_lines=osc_lines_row+osc_lines_column,
        osc_twin=osc_twin,
    ))

    # Full
    plots.update(plot_connectivity_full(
        animat_options=animat_options,
        osc_names=osc_names,
        osc_labels=osc_labels,
        osc_lines=osc_lines_row+osc_lines_column,
        osc_twin=osc_twin,
    ))

    return plots
