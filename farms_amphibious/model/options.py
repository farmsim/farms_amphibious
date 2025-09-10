"""Animat options"""

from typing import List, Dict, Union
from enum import Enum
from functools import partial
from itertools import product

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

from farms_core.options import Options
from farms_core.experiment.options import ExperimentOptions
from farms_core.simulation.options import SimulationOptions
from farms_core.model.options import (
    AnimatOptions,
    MorphologyOptions,
    LinkOptions,
    JointOptions,
    SpawnOptions,
    ControlOptions,
    MotorOptions,
    SensorsOptions,
    WaterOptions,
    ArenaOptions,
    AnimatExtensionOptions,
)
from .convention import AmphibiousConvention

# pylint: disable=too-many-lines,too-many-arguments,
# pylint: disable=too-many-locals,too-many-branches
# pylint: disable=too-many-statements,too-many-instance-attributes


def options_kwargs_float_keys():
    """Options kwargs float keys"""
    return [
        'bodylimb_none', 'bodylimb_single', 'bodylimb_overlap',
        'legs_freq_gain', 'legs_freq_bias',
        'weight_osc_body_side', 'weight_osc_body_down',
        'weight_osc_legs_internal',
        'weight_osc_legs_opposite', 'weight_osc_legs_following',
        'weight_osc_legs2body', 'weight_osc_body2legs',
        'weight_sens_stretch_freq_same',
        'weight_sens_stretch_freq_up', 'weight_sens_stretch_freq_down',
        'weight_sens_stretch_amp_same',
        'weight_sens_stretch_amp_up', 'weight_sens_stretch_amp_down',
        'weight_sens_contact_body_freq_up',
        'weight_sens_contact_body_freq_down',
        'weight_sens_contact_intralimb', 'weight_sens_contact_opposite',
        'weight_sens_contact_following', 'weight_sens_contact_diagonal',
        'weight_sens_xfrc_freq_up', 'weight_sens_xfrc_freq_down',
        'weight_sens_xfrc_amp_up', 'weight_sens_xfrc_amp_down',
        'kinematics_sampling', 'kinematics_start', 'kinematics_end',
        'muscle_alpha', 'muscle_beta', 'muscle_gamma', 'muscle_delta',
    ]


def options_kwargs_float_list_keys():
    """Options kwargs float list keys"""
    return ['drives_init', 'solref']


def options_kwargs_int_keys():
    """Options kwargs int keys"""
    return ['kinematics_time_index']


def options_kwargs_int_list_keys():
    """Options kwargs int list keys"""
    return ['kinematics_indices']


def options_kwargs_str_keys():
    """Options kwargs string keys"""
    return [
        'drive_loader',
        'drive_contact_type',
        'kinematics_file',
    ]


def options_kwargs_str_list_keys():
    """Options kwargs str list keys"""
    return ['collisions_list']


def options_kwargs_bool_keys():
    """Options kwargs bool keys"""
    return ['inanimate', 'kinematics_invert', 'kinematics_degrees']


def options_kwargs_animat_keys():
    """Options kwargs animat keys"""
    return (
        options_kwargs_float_keys()
        + options_kwargs_float_list_keys()
        + options_kwargs_int_keys()
        + options_kwargs_int_list_keys()
        + options_kwargs_str_keys()
        + options_kwargs_str_list_keys()
        + options_kwargs_bool_keys()
    )


def options_kwargs_sph_float_keys():
    """Options kwargs SPH float keys"""
    return [
        'sph_log_freq',
        'sph_spacing', 'sph_hdx',
        'sph_density_solid', 'sph_rho_fluid', 'sph_depth',
        'sph_multiplier_h', 'sph_multiplier_mass',
        'sph_multiplier_volume', 'sph_multiplier_rad_s',
        'sph_factor_solid', 'sph_co', 'sph_xsph_eps',
        'sph_alpha', 'sph_beta', 'sph_gamma',
    ]


def options_kwargs_sph_keys():
    """Options kwargs SPH keys"""
    return options_kwargs_sph_float_keys()


def options_kwargs_arena_keys():
    """Options kwargs arena keys"""
    return options_kwargs_sph_keys()


def options_kwargs_all_keys():
    """Options kwargs all keys"""
    return (
        options_kwargs_animat_keys()
        + options_kwargs_arena_keys()
    )


class DriveKind(str, Enum):  # Not using StrEnum until Python 3.10 EOL
    BRAIN_LEFT = 'brain_left'
    BRAIN_RIGHT = 'brain_right'
    SPINE_LEFT = 'spine_left'
    SPINE_RIGHT = 'spine_right'


class AmphibiousOptions(AnimatOptions):
    """Simulation options"""

    def __init__(self, sdf: str, **kwargs):
        super().__init__(
            sdf=sdf,
            spawn=SpawnOptions(**kwargs.pop('spawn')),
            morphology=AmphibiousMorphologyOptions(**kwargs.pop('morphology')),
            control=(
                KinematicsControlOptions(**kwargs.pop('control'))
                if 'kinematics_file' in kwargs['control']
                else AmphibiousControlOptions(**kwargs.pop('control'))
            ),
            extensions=[
                AnimatExtensionOptions(**extension)
                for extension in kwargs.pop('extensions')
            ],
        )
        self.show_xfrc = kwargs.pop('show_xfrc')
        self.scale_xfrc = kwargs.pop('scale_xfrc')
        self.mujoco = kwargs.pop('mujoco')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def default(cls):
        """Deafault options"""
        return cls.from_options({})

    @classmethod
    def from_options(cls, kwargs=None):
        """From options"""
        options = {}
        options['sdf'] = kwargs.pop('sdf_path')
        options['name'] = kwargs.pop('name', 'Animat')
        options['morphology'] = kwargs.pop(
            'morphology',
            AmphibiousMorphologyOptions.from_options(kwargs),
        )
        convention = AmphibiousConvention.from_morphology(
            morphology=options['morphology'],
            **{
                key: kwargs.get(key, False)
                for key in ('single_osc_body', 'single_osc_legs')
                if key in kwargs
            },
        )
        options['spawn'] = kwargs.pop(
            'spawn',
            SpawnOptions.from_options(kwargs)
        )
        options['mujoco'] = kwargs.pop('mujoco', {})
        if 'solref' in kwargs:
            options['mujoco']['solref'] = kwargs.pop('solref')
        kinematics_file = kwargs.get('kinematics_file', None)
        if 'control' in kwargs:
            options['control'] = kwargs.pop('control')
        elif kinematics_file is not None:  # Kinematics controller
            options['control'] = KinematicsControlOptions.from_options(kwargs)
            options['control'].defaults_from_convention(convention, kwargs)
        else:  # Amphibious controller
            options['control'] = AmphibiousControlOptions.from_options(kwargs)
            options['control'].defaults_from_convention(convention, kwargs)
        options['show_xfrc'] = kwargs.pop('show_xfrc', False)
        options['scale_xfrc'] = kwargs.pop('scale_xfrc', 1)
        assert not kwargs, f'Unknown kwargs: {kwargs}'
        return cls(**options)

    def state_init(self):
        """Initial states"""
        return [
            osc.initial_phase for osc in self.control.network.oscillators
        ] + [
            osc.initial_amplitude for osc in self.control.network.oscillators
        ] + [
            joint.initial[0] for joint in self.morphology.joints
        ]


class AmphibiousExperimentOptions(ExperimentOptions):
    """Amphibious experiment options"""

    def __init__(
            self,
            simulation: str | SimulationOptions,
            animats: list[str] | list[AmphibiousOptions],
            arenas: list[str] | list[ArenaOptions],
    ):
        super().__init__(
            simulation=simulation,
            animats=animats,
            arenas=arenas,
        )


class AmphibiousMorphologyOptions(MorphologyOptions):
    """Amphibious morphology options"""

    def __init__(self, **kwargs):
        super().__init__(
            links=[
                AmphibiousLinkOptions(**link)
                for link in kwargs.pop('links')
            ],
            self_collisions=kwargs.pop('self_collisions'),
            joints=[
                JointOptions(**joint)
                for joint in kwargs.pop('joints')
            ],
        )
        self.n_joints_body = kwargs.pop('n_joints_body')
        self.n_dof_legs = kwargs.pop('n_dof_legs')
        self.n_legs = kwargs.pop('n_legs')
        self.n_joints_passive = kwargs.pop('n_joints_passive')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        for kwarg in [
                'n_joints_body', 'n_dof_legs', 'n_legs',
                'links_names', 'joints_names',
        ]:
            if kwarg in kwargs.copy():
                options[kwarg] = kwargs.pop(kwarg)
        convention = AmphibiousConvention(**options)
        default_lateral_friction = kwargs.pop('default_lateral_friction', 1)
        # Feet handling
        feet_links = kwargs.pop('feet_links', None)
        if feet_links is None:
            feet_links = convention.feet_links_names()
        else:  # Feet defined and to be attributed
            feet_indices = [
                convention.leglink2index(
                    leg_i=leg_i,
                    side_i=side_i,
                    joint_i=convention.n_dof_legs-1,
                )
                for leg_i in range(convention.n_legs//2)
                for side_i in range(2)
            ]
            assert len(feet_indices) == len(feet_links), (
                f'len({feet_indices}) != len({feet_links})'
            )
            for index, name in zip(feet_indices, feet_links):
                convention.links_names[index] = name
        # Links and joints
        links_names = convention.links_names
        joints_names = convention.joints_names
        options.pop('links_names', None)
        options.pop('joints_names', None)
        # Feet friction
        feet_friction = kwargs.pop('feet_friction', None)
        if feet_friction is None:
            feet_friction = default_lateral_friction
        if isinstance(feet_friction, (float, int)):
            feet_friction = [feet_friction]*convention.n_legs_pair()
        elif len(feet_friction) < convention.n_legs_pair():
            feet_friction += [feet_friction[0]]*(
                convention.n_legs_pair() - len(feet_friction)
            )
        links_friction_lateral = kwargs.pop(
            'links_friction_lateral',
            [
                feet_friction[feet_links.index(link)//2]
                if link in feet_links
                else default_lateral_friction
                for link in links_names
            ],
        )
        links_friction_spinning = kwargs.pop(
            'links_friction_spinning',
            [0 for link in links_names],
        )
        links_friction_rolling = kwargs.pop(
            'links_friction_rolling',
            [0 for link in links_names],
        )
        links_no_collisions = kwargs.pop('links_no_collisions', (
            [
                convention.bodylink2name(body_i)
                for body_i in range(1, options['n_joints_body'])
            ] + [
                convention.leglink2name(leg_i, side_i, joint_i)
                for leg_i in range(options['n_legs']//2)
                for side_i in range(2)
                for joint_i in range(options['n_dof_legs']-1)
            ] if kwargs.pop('reduced_collisions', False) else []
        ))
        links_linear_damping = kwargs.pop(
            'links_linear_damping',
            [0 for link in links_names],
        )
        links_angular_damping = kwargs.pop(
            'links_angular_damping',
            [0 for link in links_names],
        )
        default_restitution = kwargs.pop('default_restitution', 0)
        links_restitution = kwargs.pop(
            'links_restitution',
            [default_restitution for link in links_names],
        )
        links_density = kwargs.pop('density', None)
        links_swimming = kwargs.pop('links_swimming', links_names)
        links_mass_multiplier = kwargs.pop('mass_multiplier', 1)
        drag_coefficients = kwargs.pop(
            'drag_coefficients',
            [None for name in links_names],
        )
        assert (
            len(links_names)
            == len(links_friction_lateral)
            == len(links_friction_spinning)
            == len(links_friction_rolling)
            == len(drag_coefficients)
        ), (
            'links_name,'
            ' links_friction_lateral,'
            ' links_friction_spinning,'
            ' links_friction_rolling,'
            ' drag_coefficients',
            np.shape(links_names),
            np.shape(links_friction_lateral),
            np.shape(links_friction_spinning),
            np.shape(links_friction_rolling),
            np.shape(drag_coefficients),
            links_names,
        )
        options['links'] = kwargs.pop(
            'links',
            [
                AmphibiousLinkOptions(
                    name=name,
                    collisions=name not in links_no_collisions,
                    density=links_density,
                    mass_multiplier=links_mass_multiplier,
                    swimming=name in links_swimming,
                    drag_coefficients=drag,
                    friction=[lateral, spin, roll],
                    extras={
                        'restitution': restitution,
                        'linearDamping': linear,
                        'angularDamping': angular,
                    },
                )
                for (
                        name, lateral, spin, roll,
                        drag, linear, angular, restitution
                ) in zip(
                    links_names,
                    links_friction_lateral,
                    links_friction_spinning,
                    links_friction_rolling,
                    drag_coefficients,
                    links_linear_damping,
                    links_angular_damping,
                    links_restitution,
                )
            ]
        )
        options['self_collisions'] = kwargs.pop('self_collisions', [])
        joints_positions = kwargs.pop(
            'joints_positions',
            [0 for name in joints_names]
        )
        joints_velocities = kwargs.pop(
            'joints_velocities',
            [0 for name in joints_names]
        )
        joints_stiffness = kwargs.pop(
            'joints_stiffness',
            [0 for name in joints_names]
        )
        joints_damping = kwargs.pop(
            'joints_damping',
            [0 for name in joints_names]
        )
        max_velocity = kwargs.pop('max_velocity', np.inf)
        if 'joints' not in kwargs:
            assert all(len(element) == len(joints_names) for element in (
                joints_positions,
                joints_velocities,
                joints_stiffness,
                joints_damping,
            )), (
                'Not all same size:'
                f' position: {len(joints_positions)},'
                f' velocity: {len(joints_velocities)},'
                f' stiffness: {len(joints_stiffness)}'
                f' damping: {len(joints_damping)}'
            )
        options['joints'] = kwargs.pop(
            'joints',
            [
                JointOptions(
                    name=name,
                    initial=[position, velocity],
                    stiffness=stiffness,
                    springref=0,
                    damping=damping,
                    limits=[
                        [-np.inf, np.inf],
                        [-max_velocity, max_velocity],
                    ],
                    extras={},
                )
                for name, position, velocity, stiffness, damping in zip(
                    joints_names,
                    joints_positions,
                    joints_velocities,
                    joints_stiffness,
                    joints_damping,
                )
            ]
        )
        morphology = cls(**options)
        if kwargs.pop('use_self_collisions', False):
            convention = AmphibiousConvention.from_morphology(morphology)
            morphology.self_collisions += [
                # Body-body collisions
                [
                    convention.bodylink2name(body0),
                    convention.bodylink2name(body1),
                ]
                for body0 in range(options['n_joints_body']+1)
                for body1 in range(options['n_joints_body']+1)
                if abs(body1 - body0) > 2  # Avoid neighbouring collisions
            ] + [
                # Body-leg collisions
                [
                    convention.bodylink2name(body0),
                    convention.leglink2name(leg_i, side_i, joint_i),
                ]
                for body0 in range(options['n_joints_body']+1)
                for leg_i in range(options['n_legs']//2)
                for side_i in range(2)
                for joint_i in [options['n_dof_legs']-1]  # End-effector
            ] + [
                # Leg-leg collisions
                [
                    convention.leglink2name(leg0, side0, joint0),
                    convention.leglink2name(leg1, side1, joint1),
                ]
                for leg0 in range(options['n_legs']//2)
                for leg1 in range(options['n_legs']//2)
                for side0 in range(2)
                for side1 in range(2)
                for joint0 in [options['n_dof_legs']-1]
                for joint1 in [options['n_dof_legs']-1]
                if leg0 != leg1 or side0 != side1 or joint0 != joint1
            ]
            for links in morphology.self_collisions:
                assert links[0] != links[1], f'Collision to self: {links}'
        collisions_list = kwargs.pop('collisions_list', [])
        if collisions_list:
            morphology.self_collisions += [
                [
                    collisions_list[2*i+0],
                    collisions_list[2*i+1],
                ]
                for i in range(len(collisions_list)//2)
            ]
        return morphology

    def n_joints_legs(self):
        """Number of legs joints"""
        return self.n_legs*self.n_dof_legs


class AmphibiousLinkOptions(LinkOptions):
    """Amphibious link options"""

    def __init__(self, **kwargs):
        super().__init__(
            name=kwargs.pop('name'),
            collisions=kwargs.pop('collisions'),
            friction=kwargs.pop('friction'),
            extras=kwargs.pop('extras', {}),
        )
        self.density = kwargs.pop('density')
        self.swimming = kwargs.pop('swimming')
        self.drag_coefficients = kwargs.pop('drag_coefficients')
        self.mass_multiplier: float = kwargs.pop('mass_multiplier')
        assert not kwargs, f'Unknown kwargs: {kwargs}'




class AmphibiousControlOptions(ControlOptions):
    """Amphibious control options"""

    def __init__(self, **kwargs):
        super().__init__(
            controller_loader=kwargs.pop(
                'controller_loader',
                'farms_amphibious.control.amphibious.AmphibiousController',
            ),
            sensors=AmphibiousSensorsOptions(**kwargs.pop('sensors')),
            motors=[
                AmphibiousMotorOptions(**motor)
                for motor in kwargs.pop('motors')
            ],
        )
        network_options = kwargs.pop('network', None)
        self.network = (
            AmphibiousNetworkOptions(**network_options)
            if network_options is not None
            and 'oscillators' in network_options
            else None
        )
        self.muscles = [
            AmphibiousMuscleSetOptions(**muscle)
            for muscle in kwargs.pop('muscles')
        ]
        self.hill_muscles = kwargs.pop('hill_muscles', [])
        self.adhesions = [
            AmphibiousAdhesionsOptions(**adhesion)
            for adhesion in kwargs.pop('adhesions')
        ]
        self.visuals = [
            AmphibiousVisualsOptions(**visual)
            for visual in kwargs.pop('visuals')
        ]
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def options_from_kwargs(cls, kwargs):
        """Options from kwargs"""
        options = super(cls, cls).options_from_kwargs({
            'sensors': kwargs.pop(
                'sensors',
                AmphibiousSensorsOptions.options_from_kwargs(kwargs),
            ),
            'motors': kwargs.pop('motors', {}),
        })
        options['network'] = kwargs.pop(
            'network',
            AmphibiousNetworkOptions.from_options(kwargs).to_dict()
        )
        options['muscles'] = kwargs.pop('muscles', [])
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""
        self.sensors.defaults_from_convention(convention, kwargs)
        self.network.defaults_from_convention(convention, kwargs)

        # Joints
        n_joints = convention.n_joints()
        offsets = [None]*n_joints

        # Motor gains
        motor_gains = kwargs.pop('motor_gains', [[0]]*n_joints)

        # Turning body
        for joint_i in range(convention.n_joints_body):
            for side_i in range(2):
                offsets[convention.bodyjoint2index(joint_i=joint_i)] = (
                    AmphibiousMotorOffsetOptions(
                        gain=0,
                        bias=0,
                        low=1,
                        high=5,
                        saturation=0,
                        rate=2,
                    )
                )

        # Turning legs
        legs_offsets_walking = kwargs.pop(
            'legs_offsets_walking',
            [0]*convention.n_dof_legs
        )
        legs_offsets_swimming = kwargs.pop(
            'legs_offsets_swimming',
            [0]*convention.n_dof_legs
        )
        leg_turn_gain = kwargs.pop(
            'leg_turn_gain',
            [0, 0]
            if convention.n_legs == 4
            else (-np.ones(convention.n_legs_pair())).tolist()
        )
        leg_side_turn_gain = kwargs.pop(
            'leg_side_turn_gain',
            [0, 0]
        )
        leg_joint_turn_gain = kwargs.pop(
            'leg_joint_turn_gain',
            [0]*convention.n_dof_legs
        )

        # Augment parameters
        repeat = partial(np.repeat, repeats=convention.n_legs_pair(), axis=0)
        if np.ndim(legs_offsets_walking) == 1:
            legs_offsets_walking = repeat([legs_offsets_walking]).tolist()
        if np.ndim(legs_offsets_swimming) == 1:
            legs_offsets_swimming = repeat([legs_offsets_swimming]).tolist()
        if np.ndim(leg_side_turn_gain) == 1:
            leg_side_turn_gain = repeat([leg_side_turn_gain]).tolist()
        if np.ndim(leg_joint_turn_gain) == 1:
            leg_joint_turn_gain = repeat([leg_joint_turn_gain]).tolist()

        # Motors offsets for walking and swimming
        for leg_i in range(convention.n_legs_pair()):
            for side_i in range(2):
                for joint_i in range(convention.n_dof_legs):
                    offsets[convention.legjoint2index(
                        leg_i=leg_i,
                        side_i=side_i,
                        joint_i=joint_i,
                    )] = AmphibiousMotorOffsetOptions(
                        gain=(
                            leg_turn_gain[leg_i]
                            * leg_side_turn_gain[leg_i][side_i]
                            * leg_joint_turn_gain[leg_i][joint_i]
                        ),
                        bias=legs_offsets_walking[leg_i][joint_i],
                        low=1,
                        high=3,
                        saturation=legs_offsets_swimming[leg_i][joint_i],
                        rate=2,
                    )

        # Amphibious joints control
        if not self.motors:
            self.motors = [
                AmphibiousMotorOptions(
                    joint_name=None,
                    control_types=[],
                    limits_torque=None,
                    gains=None,
                    equation=None,
                    transform=AmphibiousMotorTransformOptions(
                        gain=None,
                        bias=None,
                    ),
                    offsets=AmphibiousMotorOffsetOptions(
                        gain=None,
                        bias=None,
                        low=None,
                        high=None,
                        saturation=None,
                        rate=None,
                    ),
                    passive=AmphibiousPassiveJointOptions(
                        is_passive=False,
                        stiffness_coefficient=0,
                        damping_coefficient=0,
                        friction_coefficient=0,
                    ),
                )
                for joint in range(n_joints)
            ]
        joints_names = kwargs.pop(
            'joints_control_names',
            convention.joints_names,
        )
        transform_gain = kwargs.pop(
            'transform_gain',
            {joint_name: 1 for joint_name in joints_names},
        )
        transform_bias = kwargs.pop(
            'transform_bias',
            {joint_name: 0 for joint_name in joints_names},
        )
        default_max_torque = kwargs.pop('default_max_torque', np.inf)
        max_torques = kwargs.pop(
            'max_torques',
            {joint_name: default_max_torque for joint_name in joints_names},
        )
        default_equation = kwargs.pop('default_equation', 'position')
        equations = kwargs.pop(
            'equations',
            {
                joint_name: (
                    'phase'
                    if convention.single_osc_body
                    and joint_i < convention.n_joints_body
                    or convention.single_osc_legs
                    and joint_i >= convention.n_joints_body
                    else default_equation
                )
                for joint_i, joint_name in enumerate(joints_names)
            },
        )
        for motor_i, motor in enumerate(self.motors):

            # Control
            if motor.joint_name is None:
                motor.joint_name = joints_names[motor_i]
            if motor.equation is None:
                motor.equation = equations[motor.joint_name]
            if not motor.control_types:
                motor.control_types = {
                    'position': ['position'],
                    'phase': ['position'],
                    'ekeberg_muscle': ['velocity', 'torque'],
                    'ekeberg_muscle_explicit': ['torque'],
                    'passive': ['velocity', 'torque'],
                    'passive_explicit': ['torque'],
                }[motor.equation]
            if motor.limits_torque is None:
                motor.limits_torque = [
                    -max_torques[motor.joint_name],
                    +max_torques[motor.joint_name],
                ]
            if motor.gains is None:
                motor.gains = motor_gains[motor_i]

            # Transform
            if motor.transform.gain is None:
                motor.transform.gain = transform_gain[motor.joint_name]
            if motor.transform.bias is None:
                motor.transform.bias = transform_bias[motor.joint_name]

            # Offset
            if motor.offsets.gain is None:
                motor.offsets.gain = offsets[motor_i]['gain']
            if motor.offsets.bias is None:
                motor.offsets.bias = offsets[motor_i]['bias']
            if motor.offsets.low is None:
                motor.offsets.low = offsets[motor_i]['low']
            if motor.offsets.high is None:
                motor.offsets.high = offsets[motor_i]['high']
            if motor.offsets.saturation is None:
                motor.offsets.saturation = offsets[motor_i]['saturation']
            if motor.offsets.rate is None:
                motor.offsets.rate = offsets[motor_i]['rate']

        # Passive
        joints_passive = kwargs.pop('joints_passive', [])
        self.sensors.joints += [name for name, *_ in joints_passive]
        self.motors += [
            AmphibiousMotorOptions(
                joint_name=joint_name,
                control_types=['velocity', 'torque'],
                limits_torque=[-default_max_torque, default_max_torque],
                gains=None,
                equation='passive',
                transform=AmphibiousMotorTransformOptions(
                    gain=1,
                    bias=0,
                ),
                offsets=None,
                passive=AmphibiousPassiveJointOptions(
                    is_passive=True,
                    stiffness_coefficient=stiffness,
                    damping_coefficient=damping,
                    friction_coefficient=friction,
                ),
            )
            for joint_name, stiffness, damping, friction in joints_passive
        ]

        # Muscles
        if not self.muscles:
            self.muscles = [
                AmphibiousMuscleSetOptions(
                    joint_name=None,
                    osc1=None,
                    osc2=None,
                    alpha=None,
                    beta=None,
                    gamma=None,
                    delta=None,
                    epsilon=None,
                )
                for joint_i in range(n_joints)
            ]
        default_alpha = kwargs.pop('muscle_alpha', 0)
        default_beta = kwargs.pop('muscle_beta', 0)
        default_gamma = kwargs.pop('muscle_gamma', 0)
        default_delta = kwargs.pop('muscle_delta', 0)
        default_epsilon = kwargs.pop('muscle_epsilon', 0)
        for joint_i, muscle in enumerate(self.muscles):
            if muscle.joint_name is None:
                muscle.joint_name = joints_names[joint_i]
            if muscle.osc1 is None or muscle.osc2 is None:
                osc_idx = convention.osc_indices(joint_i)
                assert osc_idx[0] < len(self.network.oscillators), (
                    f'{joint_i}: '
                    f'{osc_idx[0]} !< {len(self.network.oscillators)}'
                )
                muscle.osc1 = self.network.oscillators[osc_idx[0]].name
                if len(osc_idx) > 1:
                    assert osc_idx[1] < len(self.network.oscillators), (
                        f'{joint_i}: '
                        f'{osc_idx[1]} !< {len(self.network.oscillators)}'
                    )
                    muscle.osc2 = self.network.oscillators[osc_idx[1]].name
            if muscle.alpha is None:
                muscle.alpha = default_alpha
            if muscle.beta is None:
                muscle.beta = default_beta
            if muscle.gamma is None:
                muscle.gamma = default_gamma
            if muscle.delta is None:
                muscle.delta = default_delta
            if muscle.epsilon is None:
                muscle.epsilon = default_epsilon

    def motors_offsets(self):
        """Motors offsets"""
        return [
            {
                key: getattr(motor.offsets, key)
                for key in [
                        'gain', 'bias',
                        'low', 'high',
                        'saturation_low', 'saturation_high',
                ]
            }
            for motor in self.motors
            if motor.offsets is not None
        ]

    def motors_offset_rates(self):
        """Motors rates"""
        return [
            motor.offsets.rate
            for motor in self.motors
            if motor.offsets is not None
        ]

    def motors_transform_gain(self):
        """Motors gain amplitudes"""
        return [motor.transform.gain for motor in self.motors]

    def motors_transform_bias(self):
        """Motors offset bias"""
        return [motor.transform.bias for motor in self.motors]

    def drives_contacts_indices(self):
        """Drives contacts indices"""
        for drive in self.network.drives:
            for contact in drive.contacts:
                assert contact in self.sensors.contacts, (
                    f'{contact=} not in {self.sensors.contacts=}'
                )
        return [
            [self.sensors.contacts.index(contact) for contact in drive.contacts]
            for drive in self.network.drives
        ]


class KinematicsControlOptions(ControlOptions):
    """Amphibious kinematics control options"""

    def __init__(self, **kwargs):
        super().__init__(
            sensors=(AmphibiousSensorsOptions(**kwargs.pop('sensors'))),
            motors=[
                AmphibiousMotorOptions(**motor)
                for motor in kwargs.pop('motors')
            ],
            muscles=kwargs.pop('muscles', []),
        )
        self.hill_muscles = kwargs.pop('hill_muscles', [])
        self.kinematics_file = kwargs.pop('kinematics_file')
        self.kinematics_sampling = kwargs.pop('kinematics_sampling')
        self.kinematics_indices = kwargs.pop('kinematics_indices')
        self.kinematics_time_index = kwargs.pop('kinematics_time_index')
        self.kinematics_invert = kwargs.pop('kinematics_invert')
        self.kinematics_degrees = kwargs.pop('kinematics_degrees')
        self.kinematics_start = kwargs.pop('kinematics_start')
        self.kinematics_end = kwargs.pop('kinematics_end')
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def options_from_kwargs(cls, kwargs):
        """Options from kwargs"""
        options = super(cls, cls).options_from_kwargs({
            'sensors': kwargs.pop(
                'sensors',
                AmphibiousSensorsOptions.options_from_kwargs(kwargs),
            ),
            'motors': kwargs.pop('motors', {}),
        })
        options['kinematics_file'] = kwargs.pop('kinematics_file', '')
        options['kinematics_sampling'] = kwargs.pop('kinematics_sampling', 0)
        options['kinematics_indices'] = kwargs.pop('kinematics_indices', None)
        options['kinematics_time_index'] = kwargs.pop('kinematics_time_index', None)
        options['kinematics_degrees'] = kwargs.pop('kinematics_degrees', False)
        options['kinematics_invert'] = kwargs.pop('kinematics_invert', False)
        options['kinematics_start'] = kwargs.pop('kinematics_start', 0)
        options['kinematics_end'] = kwargs.pop('kinematics_end', 0)
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""
        self.sensors.defaults_from_convention(convention, kwargs)

        # Joints
        n_joints = convention.n_joints()
        offsets = [None]*n_joints

        # Motor gains
        motor_gains = kwargs.pop('motor_gains', [[0]]*n_joints)

        # Turning body
        for joint_i in range(convention.n_joints_body):
            for side_i in range(2):
                offsets[convention.bodyjoint2index(joint_i=joint_i)] = (
                    AmphibiousMotorOffsetOptions(
                        gain=0,
                        bias=0,
                        low=1,
                        high=5,
                        saturation=0,
                        rate=2,
                    )
                )

        # Turning legs
        legs_offsets_walking = kwargs.pop(
            'legs_offsets_walking',
            [0]*convention.n_dof_legs,
        )
        legs_offsets_swimming = kwargs.pop(
            'legs_offsets_swimming',
            [0]*convention.n_dof_legs,
        )
        leg_turn_gain = kwargs.pop(
            'leg_turn_gain',
            [0, 0]
            if convention.n_legs == 4
            else (-np.ones(convention.n_legs_pair())).tolist(),
        )
        leg_side_turn_gain = kwargs.pop(
            'leg_side_turn_gain',
            [0, 0],
        )
        leg_joint_turn_gain = kwargs.pop(
            'leg_joint_turn_gain',
            [0]*convention.n_dof_legs,
        )

        # Augment parameters
        repeat = partial(np.repeat, repeats=convention.n_legs_pair(), axis=0)
        if np.ndim(legs_offsets_walking) == 1:
            legs_offsets_walking = repeat([legs_offsets_walking]).tolist()
        if np.ndim(legs_offsets_swimming) == 1:
            legs_offsets_swimming = repeat([legs_offsets_swimming]).tolist()
        if np.ndim(leg_side_turn_gain) == 1:
            leg_side_turn_gain = repeat([leg_side_turn_gain]).tolist()
        if np.ndim(leg_joint_turn_gain) == 1:
            leg_joint_turn_gain = repeat([leg_joint_turn_gain]).tolist()

        # Motors offsets for walking and swimming
        for leg_i in range(convention.n_legs_pair()):
            for side_i in range(2):
                for joint_i in range(convention.n_dof_legs):
                    offsets[convention.legjoint2index(
                        leg_i=leg_i,
                        side_i=side_i,
                        joint_i=joint_i,
                    )] = AmphibiousMotorOffsetOptions(
                        gain=(
                            leg_turn_gain[leg_i]
                            * leg_side_turn_gain[leg_i][side_i]
                            * leg_joint_turn_gain[leg_i][joint_i]
                        ),
                        bias=legs_offsets_walking[leg_i][joint_i],
                        low=1,
                        high=3,
                        saturation=legs_offsets_swimming[leg_i][joint_i],
                        rate=2,
                    )

        # Amphibious joints control
        if not self.motors:
            self.motors = [
                AmphibiousMotorOptions(
                    joint_name=None,
                    control_types=[],
                    limits_torque=None,
                    gains=None,
                    equation=None,
                    transform=AmphibiousMotorTransformOptions(
                        gain=None,
                        bias=None,
                    ),
                    offsets=AmphibiousMotorOffsetOptions(
                        gain=None,
                        bias=None,
                        low=None,
                        high=None,
                        saturation=None,
                        rate=None,
                    ),
                    passive=AmphibiousPassiveJointOptions(
                        is_passive=False,
                        stiffness_coefficient=0,
                        damping_coefficient=0,
                        friction_coefficient=0,
                    ),
                )
                for joint in range(n_joints)
            ]
        joints_names = kwargs.pop(
            'joints_control_names',
            convention.joints_names,
        )
        transform_gain = kwargs.pop(
            'transform_gain',
            {joint_name: 1 for joint_name in joints_names},
        )
        transform_bias = kwargs.pop(
            'transform_bias',
            {joint_name: 0 for joint_name in joints_names},
        )
        default_max_torque = kwargs.pop('default_max_torque', np.inf)
        max_torques = kwargs.pop(
            'max_torques',
            {joint_name: default_max_torque for joint_name in joints_names},
        )
        default_equation = kwargs.pop('default_equation', 'position')
        equations = kwargs.pop(
            'equations',
            {
                joint_name: (
                    'phase'
                    if convention.single_osc_body
                    and joint_i < convention.n_joints_body
                    or convention.single_osc_legs
                    and joint_i >= convention.n_joints_body
                    else default_equation
                )
                for joint_i, joint_name in enumerate(joints_names)
            },
        )
        for motor_i, motor in enumerate(self.motors):

            # Control
            if motor.joint_name is None:
                motor.joint_name = joints_names[motor_i]
            if motor.equation is None:
                motor.equation = equations[motor.joint_name]
            if not motor.control_types:
                motor.control_types = {
                    'position': ['position'],
                }[motor.equation]
            if motor.limits_torque is None:
                motor.limits_torque = [
                    -max_torques[motor.joint_name],
                    +max_torques[motor.joint_name],
                ]
            if motor.gains is None:
                motor.gains = motor_gains[motor_i]

            # Transform
            if motor.transform.gain is None:
                motor.transform.gain = transform_gain[motor.joint_name]
            if motor.transform.bias is None:
                motor.transform.bias = transform_bias[motor.joint_name]

            # Offset
            if motor.offsets.gain is None:
                motor.offsets.gain = offsets[motor_i]['gain']
            if motor.offsets.bias is None:
                motor.offsets.bias = offsets[motor_i]['bias']
            if motor.offsets.low is None:
                motor.offsets.low = offsets[motor_i]['low']
            if motor.offsets.high is None:
                motor.offsets.high = offsets[motor_i]['high']
            if motor.offsets.saturation is None:
                motor.offsets.saturation = offsets[motor_i]['saturation']
            if motor.offsets.rate is None:
                motor.offsets.rate = offsets[motor_i]['rate']

        # Passive
        joints_passive = kwargs.pop('joints_passive', [])
        self.sensors.joints += [name for name, *_ in joints_passive]
        self.motors += [
            AmphibiousMotorOptions(
                joint_name=joint_name,
                control_types=['velocity', 'torque'],
                limits_torque=[-default_max_torque, default_max_torque],
                equation='passive',
                transform=AmphibiousMotorTransformOptions(
                    gain=1,
                    bias=0,
                ),
                offsets=None,
                passive=AmphibiousPassiveJointOptions(
                    is_passive=True,
                    stiffness_coefficient=stiffness,
                    damping_coefficient=damping,
                    friction_coefficient=friction,
                ),
            )
            for joint_name, stiffness, damping, friction in joints_passive
        ]

    def motors_transform_gain(self):
        """Motors gain amplitudes"""
        return [motor.transform.gain for motor in self.motors]

    def motors_transform_bias(self):
        """Motors offset bias"""
        return [motor.transform.bias for motor in self.motors]


class AmphibiousMotorOptions(MotorOptions):
    """Amphibious motor options"""

    def __init__(self, **kwargs):
        super().__init__(
            joint_name=kwargs.pop('joint_name'),
            control_types=kwargs.pop('control_types'),
            limits_torque=kwargs.pop('limits_torque'),
            gains=kwargs.pop('gains'),
        )
        self.equation: str = kwargs.pop('equation')
        transform = kwargs.pop('transform')
        self.transform: AmphibiousMotorTransformOptions = (
            AmphibiousMotorTransformOptions(**transform)
            if transform is not None
            else None
        )
        offsets = kwargs.pop('offsets')
        self.offsets: AmphibiousMotorOffsetOptions = (
            AmphibiousMotorOffsetOptions(**offsets)
            if offsets is not None
            else None
        )
        passive = kwargs.pop('passive')
        self.passive: AmphibiousPassiveJointOptions = (
            AmphibiousPassiveJointOptions(**passive)
            if passive is not None
            else None
        )
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousMotorTransformOptions(Options):
    """Amphibious motor options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain: float = kwargs.pop('gain')
        self.bias: float = kwargs.pop('bias')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousMotorOffsetOptions(Options):
    """Amphibious motor options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.gain: float = kwargs.pop('gain')
        self.bias: float = kwargs.pop('bias')
        self.low: float = kwargs.pop('low')
        self.high: float = kwargs.pop('high')
        self.saturation_low: float = kwargs.pop('saturation_low')
        self.saturation_high: float = kwargs.pop('saturation_high')
        self.rate: float = kwargs.pop('rate')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousSensorsOptions(SensorsOptions):
    """Amphibious sensors options"""

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""
        self.links = kwargs.pop('sensors_links', convention.links_names)
        self.joints = kwargs.pop('sensors_joints', convention.joints_names)
        self.contacts = kwargs.pop('sensors_contacts', None)
        self.xfrc = kwargs.pop('sensors_xfrc', convention.links_names)
        if self.contacts is None:
            self.contacts = convention.feet_links_names()


class AmphibiousNetworkOptions(Options):
    """Amphibious network options"""

    def __init__(self, **kwargs):
        super().__init__()

        # Drives
        self.drive_loader = kwargs.pop('drive_loader', '')
        self.drive_config = kwargs.pop('drive_config')
        self.drives: List[AmphibiousDriveOptions] = [
            AmphibiousDriveOptions(**drive)
            for drive in kwargs.pop('drives')
        ]

        # Oscillators
        self.oscillators: List[AmphibiousOscillatorOptions] = [
            AmphibiousOscillatorOptions(**oscillator)
            for oscillator in kwargs.pop('oscillators')
        ]
        self.single_osc_body: bool = kwargs.pop('single_osc_body', False)
        self.single_osc_legs: bool = kwargs.pop('single_osc_legs', False)

        # Connections
        self.osc2osc = kwargs.pop('osc2osc', None)
        self.joint2osc = kwargs.pop('joint2osc', None)
        self.contact2osc = kwargs.pop('contact2osc', None)
        self.xfrc2osc = kwargs.pop('xfrc2osc', None)
        self.drive2osc = kwargs.pop('drive2osc', None)
        self.drive2joint = kwargs.pop('drive2joint', None)

        # Kwargs
        assert not kwargs, f'Unknown kwargs: {kwargs}'

    @classmethod
    def from_options(cls, kwargs):
        """From options"""
        options = {}
        options['drives'] = kwargs.pop('drives', [])
        options['drive_loader'] = kwargs.pop('drive_loader', '')
        options['drive_config'] = kwargs.pop('drive_config', '')
        options['oscillators'] = kwargs.pop('oscillators', [])
        options['single_osc_body'] = kwargs.pop('single_osc_body', False)
        options['single_osc_legs'] = kwargs.pop('single_osc_legs', False)
        # Connectivity
        for option in [
                'osc2osc', 'drive2osc',
                'joint2osc', 'contact2osc', 'xfrc2osc',
        ]:
            options[option] = kwargs.pop(option, None)
        return cls(**options)

    def defaults_from_convention(self, convention, kwargs):
        """Defaults from convention"""

        # Parameters
        legs_amplitudes = kwargs.pop(
            'legs_amplitudes',
            [0]*convention.n_dof_legs,
        )

        # Augment parameters
        repeat = partial(np.repeat, repeats=convention.n_legs_pair(), axis=0)
        if np.ndim(legs_amplitudes) == 1:
            legs_amplitudes = repeat([legs_amplitudes]).tolist()

        # Oscillators
        n_oscillators = convention.n_osc()
        if not self.oscillators:
            self.oscillators = [
                AmphibiousOscillatorOptions(
                    name=None,
                    initial_phase=None,
                    initial_amplitude=None,
                    frequency_gain=None,
                    frequency_bias=None,
                    frequency_low=None,
                    frequency_high=None,
                    frequency_saturation=None,
                    amplitude_gain=None,
                    amplitude_bias=None,
                    amplitude_low=None,
                    amplitude_high=None,
                    amplitude_saturation=None,
                    rate=None,
                    modular_phase=None,
                    modular_amplitude=None,
                )
                for osc_i in range(n_oscillators)
            ]
        state_init_smart = kwargs.pop('state_init_smart', False)
        random_state = RandomState(MT19937(SeedSequence(123456789)))
        state_init = kwargs.pop(
            'state_init',
            self.default_state_init(convention).tolist()
            if state_init_smart
            else np.concatenate([
                    # Phases
                    2*np.pi*random_state.rand(convention.n_osc()),
                    # Amplitudes
                    np.zeros(convention.n_osc()),
                    # Joints
                    np.zeros(convention.n_joints()),
            ]),
        )
        assert len(state_init) == convention.n_states()
        osc_frequencies = kwargs.pop(
            'osc_frequencies',
            self.default_osc_frequencies(convention, kwargs),
        )
        osc_amplitudes = kwargs.pop(
            'osc_amplitudes',
            self.default_osc_amplitudes(
                convention,
                body_walk_amplitude=kwargs.pop('body_walk_amplitude', 0),
                body_osc_gain=kwargs.pop('body_osc_gain', 0),
                body_osc_bias=kwargs.pop('body_osc_bias', 0),
                legs_amplitudes=legs_amplitudes,
                legs_osc_gain=kwargs.pop('legs_osc_gain', 0),
                legs_osc_bias=kwargs.pop('legs_osc_bias', 0),
            )
        )
        osc_rates = kwargs.pop('osc_rates', self.default_osc_rates(convention))
        osc_modular_phases = kwargs.pop(
            'osc_modular_phases',
            self.default_osc_modular_phases(
                convention=convention,
                phases=kwargs.pop('modular_phases', np.zeros(5)),
            )
        )
        osc_modular_amplitudes = kwargs.pop(
            'osc_modular_amplitudes',
            self.default_osc_modular_amplitudes(
                convention=convention,
                amplitudes=kwargs.pop('modular_amplitudes', np.zeros(5)),
            )
        )
        for osc_i, osc in enumerate(self.oscillators):
            if osc.name is None:
                osc.name = convention.oscindex2name(osc_i)
            if osc.initial_phase is None:
                osc.initial_phase = float(state_init[osc_i])
            if osc.initial_amplitude is None:
                osc.initial_amplitude = float(state_init[osc_i+n_oscillators])
            if osc.frequency_gain is None:
                osc.frequency_gain = osc_frequencies[osc_i]['gain']
            if osc.frequency_bias is None:
                osc.frequency_bias = osc_frequencies[osc_i]['bias']
            if osc.frequency_low is None:
                osc.frequency_low = osc_frequencies[osc_i]['low']
            if osc.frequency_high is None:
                osc.frequency_high = osc_frequencies[osc_i]['high']
            if osc.frequency_saturation is None:
                osc.frequency_saturation = osc_frequencies[osc_i]['saturation']
            if osc.amplitude_gain is None:
                osc.amplitude_gain = osc_amplitudes[osc_i]['gain']
            if osc.amplitude_bias is None:
                osc.amplitude_bias = osc_amplitudes[osc_i]['bias']
            if osc.amplitude_low is None:
                osc.amplitude_low = osc_amplitudes[osc_i]['low']
            if osc.amplitude_high is None:
                osc.amplitude_high = osc_amplitudes[osc_i]['high']
            if osc.amplitude_saturation is None:
                osc.amplitude_saturation = osc_amplitudes[osc_i]['saturation']
            if osc.rate is None:
                osc.rate = osc_rates[osc_i]
            if osc.modular_phase is None:
                osc.modular_phase = osc_modular_phases[osc_i]
            if osc.modular_amplitude is None:
                osc.modular_amplitude = osc_modular_amplitudes[osc_i]

        # Connectivity
        bodylimb_none = kwargs.pop('bodylimb_none', False)
        bodylimb_single = kwargs.pop('bodylimb_single', False)
        bodylimb_overlap = kwargs.pop('bodylimb_overlap', True)
        if self.osc2osc is None:
            # body_stand_shift = kwargs.pop('body_stand_shift', pi2)
            n_leg_pairs = convention.n_legs_pair()
            legs_splits = np.array_split(
                np.arange(convention.n_joints_body),
                n_leg_pairs
            ) if convention.n_legs and convention.n_joints_body else None
            empty = np.array([], dtype=int)
            default_legbodyconnections = [  # Body osc to connect to each limb
                []
                if bodylimb_none
                else [split[int((len(split) - 1)/2)]]
                if bodylimb_single
                else np.concatenate([
                    [split[0]-1]
                    if split_i
                    and len(split) <= len(legs_splits[split_i-1])
                    else empty,
                    split,
                    [split[-1]+1]
                    if split_i < (n_leg_pairs-1)
                    and len(split) <= len(legs_splits[split_i+1])
                    else empty,
                ])
                if bodylimb_overlap
                else split
                for split_i, split in enumerate(legs_splits)
            ] if legs_splits is not None else []
            standing = kwargs.pop('standing_wave', True)
            repeat = partial(np.repeat, repeats=n_leg_pairs, axis=0)
            intralimb_phases = kwargs.pop(
                'intralimb_phases',
                [0]*convention.n_dof_legs,
            )
            if np.ndim(intralimb_phases) == 1:
                intralimb_phases = repeat([intralimb_phases]).tolist()
            self.osc2osc = (
                self.default_osc2osc(
                    convention=convention,
                    weight_body2body_down=kwargs.pop(
                        'weight_osc_body_down',
                        0,
                    ),
                    weight_body2body_side=kwargs.pop(
                        'weight_osc_body_side',
                        0,
                    ),
                    phase_body2body=kwargs.pop(
                        'body_phase_bias',
                        2*np.pi/convention.n_joints_body
                        if convention.n_joints_body > 0
                        else 0
                    ),
                    weight_intralimb=kwargs.pop(
                        'weight_osc_legs_internal',
                        0,
                    ),
                    weight_interlimb_opposite=kwargs.pop(
                        'weight_osc_legs_opposite',
                        0,
                    ),
                    weight_interlimb_following=kwargs.pop(
                        'weight_osc_legs_following',
                        0,
                    ),
                    weight_limb2body=kwargs.pop(
                        'weight_osc_legs2body',
                        0,
                    ),
                    weight_body2limb=kwargs.pop(
                        'weight_osc_body2legs',
                        0,
                    ),
                    intralimb_phases=intralimb_phases,
                    phase_limb_follow=kwargs.pop(
                        'leg_phase_follow',
                        np.pi,
                    ),
                    body_walk_phases=kwargs.pop(
                        'body_walk_phases',
                        # [
                        #     body_i*2*np.pi/convention.n_joints_body
                        #     + body_stand_shift
                        #     for body_i in range(convention.n_joints_body)
                        # ]
                        np.concatenate([
                            np.full(len(split), (np.pi*(split_i+1)) % (2*np.pi))
                            for split_i, split in enumerate(legs_splits)
                        ]).tolist()
                        if standing
                        and legs_splits is not None
                        else np.zeros(convention.n_joints_body)
                        if legs_splits is not None
                        else [],
                    ),
                    legbodyjoints=kwargs.pop(
                        'legbodyjoints',
                        range(1)
                        if kwargs.pop('reduced_limb_body', True)
                        else range(convention.n_dof_legs-1),
                    ),
                    legbodyconnections=kwargs.pop(
                        'legbodyconnections',
                        [
                            range(convention.n_joints_body)
                            for leg_i in range(n_leg_pairs)
                        ]
                        if kwargs.pop('full_leg_body', False)
                        else default_legbodyconnections,
                    ),
                    standing=standing,
                )
            )
        if self.joint2osc is None:
            self.joint2osc = self.default_joint2osc(
                convention,
                kwargs.pop('weight_sens_stretch_freq_up', 0),
                kwargs.pop('weight_sens_stretch_freq_same', 0),
                kwargs.pop('weight_sens_stretch_freq_down', 0),
                kwargs.pop('weight_sens_stretch_amp_up', 0),
                kwargs.pop('weight_sens_stretch_amp_same', 0),
                kwargs.pop('weight_sens_stretch_amp_down', 0),
            )
        if self.contact2osc is None:
            self.contact2osc = self.default_contact2osc(
                convention,
                kwargs.pop('weight_sens_contact_body_freq_up', 0),
                kwargs.pop('weight_sens_contact_body_freq_down', 0),
                kwargs.pop('weight_sens_contact_intralimb', 0),
                kwargs.pop('weight_sens_contact_opposite', 0),
                kwargs.pop('weight_sens_contact_following', 0),
                kwargs.pop('weight_sens_contact_diagonal', 0),
            )
        if self.xfrc2osc is None:
            self.xfrc2osc = self.default_xfrc2osc(
                convention,
                kwargs.pop('weight_sens_xfrc_freq_up', 0),
                kwargs.pop('weight_sens_xfrc_freq_down', 0),
                kwargs.pop('weight_sens_xfrc_amp_up', 0),
                kwargs.pop('weight_sens_xfrc_amp_down', 0),
            )

        # Drives
        if not self.drives:
            self.drives = [
                AmphibiousDriveOptions(
                    name=None,
                    initial_value=None,
                    kind=None,
                    contacts=None,
                )
                for _ in range(convention.n_drives())
            ]
        drives_init = kwargs.pop('drives_init', [0, 0])
        drive_contact_type = kwargs.pop('drive_contact_type', '')
        n_links_body = convention.n_links_body()
        n_legs_pair = convention.n_legs_pair()
        contacts_body = [
            (name, '')
            for name in convention.body_links_names()
        ]
        contacts_feet = [
            (name, '')
            for name in convention.feet_links_names()
        ]
        overlap = round(0.1*n_links_body)
        body_splits = [
            np.arange(
                max(0, split[0]-overlap),
                min(n_links_body, split[-1]+overlap+1)
            )
            if len(split)
            else np.arange(0, n_links_body)
            for split in np.array_split(
                    np.arange(n_links_body),
                    n_leg_pairs,
            )
        ] if n_leg_pairs else []
        overlap = round(0.1*n_legs_pair)
        limb_splits = [
            np.arange(
                max(0, index-overlap),
                min(n_legs_pair, index+overlap+1)
            )
            for index in np.arange(n_legs_pair)
        ]
        for drive_i, drive in enumerate(self.drives):
            info = convention.oscindex2information(drive_i)
            if drive.name is None:
                # drive.name = f'Drive_{drive_i}'
                osc_name = convention.oscindex2name(drive_i)
                drive.name = osc_name.replace('osc', 'drv')
            if drive.initial_value is None:
                drive.initial_value = drives_init[
                    (drive_i % 2)
                    if len(drives_init) == 2
                    else drive_i
                ]
            if drive.kind is None:
                drive.kind = (
                    info['kind']
                    if info.get('region', 'spine') == 'brain'
                    else (
                        # Body
                        DriveKind.SPINE_LEFT
                        if info['side'] == 0
                        else DriveKind.SPINE_RIGHT
                    )
                    if info['body']
                    else (
                        # Limbs
                        DriveKind.SPINE_LEFT
                        if info['side_i'] == 1
                        else DriveKind.SPINE_RIGHT
                    )
                )
            if drive.contacts is None:
                if drive_contact_type == '':
                    drive.contacts = []
                elif drive_contact_type == 'all':
                    drive.contacts = contacts_body + contacts_feet
                elif drive_contact_type == 'distributed':
                    n_joints_body = convention.n_joints_body
                    if info.get('region', 'spine') == 'brain':  # Brain drive
                        drive.contacts = []
                    elif info['body']:  # Spine drive (Body)
                        joint_i = info['joint_i']
                        leg_i = round(joint_i*(n_legs_pair-1)/(n_joints_body-1))
                        drive.contacts = (
                            contacts_body[joint_i:joint_i+2]
                            + contacts_feet[2*leg_i:2*leg_i+2]
                        )
                    else:  # Spine drive (Limbs)
                        leg_i = info['leg_i']
                        body_indices = body_splits[leg_i]
                        drive.contacts = (
                            [contacts_body[i] for i in body_indices]
                            + [
                                contact
                                for leg_ii in limb_splits[leg_i]
                                for contact in contacts_feet[2*leg_ii:2*leg_ii+2]
                            ]
                        )
                else:
                    raise Exception(f'Unknown {drive_contact_type=}')
        if self.drive2osc is None:
            self.drive2osc = list(range(n_oscillators))
        if self.drive2joint is None:
            self.drive2joint = [
                [2*i, 2*i+1]
                for i in range(convention.n_joints())
            ]

    def drives_init(self):
        """Initial drives"""
        return [drive.initial_value for drive in self.drives]

    def drives_indices_kind(self, kind):
        """Drives brain indices"""
        return [
            drive_i
            for drive_i, drive in enumerate(self.drives)
            if drive.kind == kind
        ]

    def drives_brain_left_indices(self):
        """Drives left brain indices"""
        return self.drives_indices_kind(kind=DriveKind.BRAIN_LEFT)

    def drives_brain_right_indices(self):
        """Drives right brain indices"""
        return self.drives_indices_kind(kind=DriveKind.BRAIN_RIGHT)

    def drives_left_indices(self):
        """Drives (Spinal) left indices"""
        return self.drives_indices_kind(kind=DriveKind.SPINE_LEFT)

    def drives_right_indices(self):
        """Drives (Spinal) right indices"""
        return self.drives_indices_kind(kind=DriveKind.SPINE_RIGHT)

    def n_oscillators(self):
        """Number of oscillators"""
        return len(self.oscillators)

    def osc_names(self):
        """Oscillator names"""
        return [osc.name for osc in self.oscillators]

    def osc_frequencies(self):
        """Oscillator frequencies"""
        return [
            {
                'gain': osc.frequency_gain,
                'bias': osc.frequency_bias,
                'low': osc.frequency_low,
                'high': osc.frequency_high,
                'saturation_low': osc.frequency_saturation_low,
                'saturation_high': osc.frequency_saturation_high,
            }
            for osc in self.oscillators
        ]

    def osc_amplitudes(self):
        """Oscillator amplitudes"""
        return [
            {
                'gain': osc.amplitude_gain,
                'bias': osc.amplitude_bias,
                'low': osc.amplitude_low,
                'high': osc.amplitude_high,
                'saturation_low': osc.amplitude_saturation_low,
                'saturation_high': osc.amplitude_saturation_high,
            }
            for osc in self.oscillators
        ]

    def osc_rates(self):
        """Oscillator rates"""
        return [osc.rate for osc in self.oscillators]

    def osc_modular_phases(self):
        """Oscillator modular phases"""
        return [osc.modular_phase for osc in self.oscillators]

    def osc_modular_amplitudes(self):
        """Oscillator modular amplitudes"""
        return [osc.modular_amplitude for osc in self.oscillators]

    @staticmethod
    def default_state_init(convention):
        """Default state"""
        state = np.zeros(convention.n_states())
        phases_init_body = np.linspace(2*np.pi, 0, convention.n_joints_body)
        for joint_i in range(convention.n_joints_body):
            for side_osc in range(2):
                state[convention.bodyosc2index(
                    joint_i,
                    side=side_osc,
                )] = (
                    phases_init_body[joint_i]
                    + (np.pi if side_osc else 0)
                )
        phases_init_legs = [0]*convention.n_dof_legs
        for joint_i in range(convention.n_dof_legs):
            for leg_i in range(convention.n_legs_pair()):
                for side_i in range(2):
                    for side in range(1 if convention.single_osc_legs else 2):
                        state[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = (
                            (0 if leg_i else np.pi)
                            + (0 if side_i else np.pi)
                            + (0 if side else np.pi)
                            + phases_init_legs[joint_i]
                        )
        state += 1e-3*np.arange(convention.n_states())
        return state

    @staticmethod
    def default_osc_frequencies(convention, kwargs):
        """Walking parameters"""
        frequencies = [None]*convention.n_osc()

        # Body
        body_freq_gain = kwargs.pop('body_freq_gain', 0.0)
        body_freq_bias = kwargs.pop('body_freq_bias', 0.0)
        for joint_i in range(convention.n_joints_body):
            for side in range(1 if convention.single_osc_body else 2):
                frequencies[convention.bodyosc2index(joint_i, side=side)] = {
                    'gain': body_freq_gain,
                    'bias': body_freq_bias,
                    'low': 1,
                    'high': 5,
                    'saturation_low': 0,
                    'saturation_high': 0,
                }

        # legs
        legs_freq_gain = kwargs.pop('legs_freq_gain', 0.0)
        legs_freq_bias = kwargs.pop('legs_freq_bias', 0.0)
        for joint_i in range(convention.n_dof_legs):
            for leg_i in range(convention.n_legs_pair()):
                for side_i in range(2):
                    for side in range(1 if convention.single_osc_legs else 2):
                        frequencies[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = {
                            'gain': legs_freq_gain,
                            'bias': legs_freq_bias,
                            'low': 1,
                            'high': 3,
                            'saturation_low': 0,
                            'saturation_high': 0,
                        }

        return frequencies

    @staticmethod
    def default_osc_amplitudes(
            convention,
            body_walk_amplitude,
            body_osc_gain,
            body_osc_bias,
            legs_amplitudes,
            legs_osc_gain,
            legs_osc_bias,
    ):
        """Walking parameters"""
        amplitudes = [None]*convention.n_osc()
        # Body ampltidudes
        for joint_i in range(convention.n_joints_body):
            for side in range(1 if convention.single_osc_body else 2):
                amplitudes[convention.bodyosc2index(joint_i, side=side)] = {
                    'gain': body_osc_gain*body_walk_amplitude,
                    'bias': body_osc_bias*body_walk_amplitude,
                    'low': 1,
                    'high': 5,
                    'saturation': 0,
                }
        # Legs ampltidudes
        for leg_i in range(convention.n_legs_pair()):
            for joint_i in range(convention.n_dof_legs):
                legs_amplitude = legs_amplitudes[leg_i][joint_i]
                for side_i in range(2):
                    for side in range(1 if convention.single_osc_legs else 2):
                        amplitudes[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = {
                            'gain': legs_osc_gain*legs_amplitude,
                            'bias': legs_osc_bias*legs_amplitude,
                            'low': 1,
                            'high': 3,
                            'saturation': 0,
                        }
        return amplitudes

    @staticmethod
    def default_osc_rates(convention):
        """Walking parameters"""
        rates = 10*np.ones(convention.n_osc())
        return rates.tolist()

    @staticmethod
    def default_osc_modular_phases(convention, phases):
        """Default"""
        values = np.zeros(convention.n_osc())
        for joint_i in range(convention.n_dof_legs):
            phase = phases[joint_i]
            for leg_i in range(convention.n_legs_pair()):
                for side_i in range(2):
                    for side in range(1 if convention.single_osc_legs else 2):
                        values[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = (
                            phase
                            # + (0 if leg_i else np.pi)
                            # + (0 if side_i else np.pi)
                            + (0 if side else np.pi)
                        )
        return values.tolist()

    @staticmethod
    def default_osc_modular_amplitudes(convention, amplitudes):
        """Default"""
        values = np.zeros(convention.n_osc())
        for joint_i in range(convention.n_dof_legs):
            amplitude = amplitudes[joint_i]
            for leg_i in range(convention.n_legs_pair()):
                for side_i in range(2):
                    for side in range(1 if convention.single_osc_legs else 2):
                        values[convention.legosc2index(
                            leg_i,
                            side_i,
                            joint_i,
                            side=side,
                        )] = amplitude
        return values.tolist()

    @staticmethod
    def default_osc2osc(
            convention,
            weight_body2body_side,
            weight_body2body_down,
            phase_body2body,
            weight_intralimb,
            weight_interlimb_opposite,
            weight_interlimb_following,
            weight_limb2body,
            weight_body2limb,
            intralimb_phases,
            phase_limb_follow,
            body_walk_phases,
            legbodyjoints,
            legbodyconnections,
            standing,
    ):
        """Default oscillators to oscillators connectivity"""
        connectivity = []
        n_body_joints = convention.n_joints_body

        # Body
        if weight_body2body_side != 0:
            # Antagonist oscillators
            if not convention.single_osc_body:
                for i, sides in product(
                        range(n_body_joints),
                        [[1, 0], [0, 1]],
                ):
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=i,
                            side=sides[0]
                        ),
                        'out': convention.bodyosc2name(
                            joint_i=i,
                            side=sides[1]
                        ),
                        'type': 'OSC2OSC',
                        'weight': weight_body2body_side,
                        'phase_bias': np.pi,
                    })
        if weight_body2body_down != 0:
            # Following oscillators
            for i, side in product(
                    range(n_body_joints-1),
                    range(1 if convention.single_osc_body else 2),
            ):
                for osc, phase in [
                        [[i+1, i], +phase_body2body],
                        [[i, i+1], -phase_body2body],
                ]:
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=osc[0],
                            side=side
                        ),
                        'out': convention.bodyosc2name(
                            joint_i=osc[1],
                            side=side
                        ),
                        'type': 'OSC2OSC',
                        'weight': weight_body2body_down,
                        'phase_bias': phase % (2*np.pi),
                    })

        # Legs (internal)
        if weight_intralimb != 0:
            for leg_i, side_i in product(
                    range(convention.n_legs_pair()),
                    range(2),
            ):
                _options = {
                    'leg_i': leg_i,
                    'side_i': side_i
                }
                # X - X
                if not convention.single_osc_legs:
                    for joint_i, sides in product(
                            range(convention.n_dof_legs),
                            [[1, 0], [0, 1]],
                    ):
                        connectivity.append({
                            'in': convention.legosc2name(
                                **_options,
                                joint_i=joint_i,
                                side=sides[0]
                            ),
                            'out': convention.legosc2name(
                                **_options,
                                joint_i=joint_i,
                                side=sides[1]
                            ),
                            'type': 'OSC2OSC',
                            'weight': weight_intralimb,
                            'phase_bias': np.pi,
                        })

                # Following
                internal_connectivity = []
                for joint_i_0, joint_i_1 in product(
                        range(convention.n_dof_legs),
                        range(convention.n_dof_legs),
                ):
                    if joint_i_0 != joint_i_1:
                        phase = (
                            intralimb_phases[leg_i][joint_i_1]
                            - intralimb_phases[leg_i][joint_i_0]
                        )
                        internal_connectivity.extend([
                            [[joint_i_1, joint_i_0], 0, phase],
                        ])
                        if not convention.single_osc_legs:
                            internal_connectivity.extend([
                                [[joint_i_1, joint_i_0], 1, phase],
                            ])
                for joints, side, phase in internal_connectivity:
                    connectivity.append({
                        'in': convention.legosc2name(
                            **_options,
                            joint_i=joints[0],
                            side=side,
                        ),
                        'out': convention.legosc2name(
                            **_options,
                            joint_i=joints[1],
                            side=side,
                        ),
                        'type': 'OSC2OSC',
                        'weight': weight_intralimb,
                        'phase_bias': phase % (2*np.pi),
                    })

        # Opposite leg interaction
        if weight_interlimb_opposite != 0:
            for leg_i in range(convention.n_legs_pair()):
                for joint_i in range(convention.n_dof_legs):
                    for side in range(1 if convention.single_osc_legs else 2):
                        _options = {
                            'joint_i': joint_i,
                            'side': side
                        }
                        for sides in [[1, 0], [0, 1]]:
                            connectivity.append({
                                'in': convention.legosc2name(
                                    leg_i=leg_i,
                                    side_i=sides[0],
                                    **_options
                                ),
                                'out': convention.legosc2name(
                                    leg_i=leg_i,
                                    side_i=sides[1],
                                    **_options
                                ),
                                'type': 'OSC2OSC',
                                'weight': weight_interlimb_opposite,
                                'phase_bias': np.pi,
                            })

        # Following leg interaction
        if weight_interlimb_following != 0:
            for leg_pre in range(convention.n_legs_pair()-1):
                for side_i in range(2):
                    for side in range(1 if convention.single_osc_legs else 2):
                        _options = {
                            'side_i': side_i,
                            'side': side,
                            'joint_i': 0,
                        }
                        for legs, phase in [
                                [[leg_pre, leg_pre+1], phase_limb_follow],
                                [[leg_pre+1, leg_pre], -phase_limb_follow],
                        ]:
                            connectivity.append({
                                'in': convention.legosc2name(
                                    leg_i=legs[0],
                                    **_options
                                ),
                                'out': convention.legosc2name(
                                    leg_i=legs[1],
                                    **_options
                                ),
                                'type': 'OSC2OSC',
                                'weight': weight_interlimb_following,
                                'phase_bias': phase % (2*np.pi),
                            })

        # Legs-body interaction
        if weight_limb2body != 0:
            for leg_i, side_i, joint_i, side_leg_osc, lateral in product(
                    range(convention.n_legs_pair()),
                    range(2),
                    legbodyjoints,
                    range(1 if convention.single_osc_legs else 2),
                    range(1),
            ):
                for body_i in legbodyconnections[leg_i]:
                    connectivity.append({
                        'in': convention.bodyosc2name(
                            joint_i=body_i,
                            side=(
                                0
                                if convention.single_osc_body
                                else (side_i+lateral) % 2
                            ),
                        ),
                        'out': convention.legosc2name(
                            leg_i=leg_i,
                            side_i=side_i,
                            joint_i=joint_i,
                            side=side_leg_osc
                        ),
                        'type': 'OSC2OSC',
                        'weight': weight_limb2body,
                        'phase_bias': (
                            body_walk_phases[body_i]
                            + np.pi*(
                                1
                                + lateral
                                + leg_i
                                + side_leg_osc
                            )
                            - intralimb_phases[leg_i][joint_i]
                        ) % (2*np.pi),
                    })

        # Body-legs interaction
        if weight_body2limb != 0:
            for leg_i, side_i, joint_i, side_leg_osc, lateral in product(
                    range(convention.n_legs_pair()),
                    range(2),
                    legbodyjoints,
                    range(1 if convention.single_osc_legs else 2),
                    range(1),
            ):
                for body_i in legbodyconnections[leg_i]:
                    connectivity.append({
                        'in': convention.legosc2name(
                            leg_i=leg_i,
                            side_i=side_i,
                            joint_i=joint_i,
                            side=side_leg_osc
                        ),
                        'out': convention.bodyosc2name(
                            joint_i=body_i,
                            side=(side_i+lateral) % 2
                        ),
                        'type': 'OSC2OSC',
                        'weight': weight_body2limb,
                        'phase_bias': (-(
                            body_walk_phases[body_i]
                            + np.pi*(
                                1
                                + lateral
                                + (leg_i if standing else 0.5)
                                + side_leg_osc
                            )
                            - intralimb_phases[leg_i][joint_i]
                        )) % (2*np.pi),
                    })
        return connectivity

    @staticmethod
    def default_joint2osc(
            convention,
            weight_frequency_up,
            weight_frequency_same,
            weight_frequency_down,
            weight_amplitude_up,
            weight_amplitude_same,
            weight_amplitude_down,
    ):
        """Default joint sensors to oscillators connectivity"""
        connectivity = []
        if weight_frequency_up:
            for joint_i, side_osc in product(
                    range(convention.n_joints_body-1),
                    range(2),
            ):
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_osc,
                    ),
                    'out': convention.bodyjoint2name(joint_i+1),
                    'type': 'STRETCH2FREQ',
                    'weight': (-1 if side_osc else 1)*weight_frequency_up,
                })
        if weight_frequency_same:
            for joint_i, side_osc in product(
                    range(convention.n_joints_body),
                    range(2),
            ):
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_osc,
                    ),
                    'out': convention.bodyjoint2name(joint_i),
                    'type': 'STRETCH2FREQ',
                    'weight': (-1 if side_osc else 1)*weight_frequency_same,
                })
        if weight_frequency_down:
            for joint_i, side_osc in product(
                    range(convention.n_joints_body-1),
                    range(2),
            ):
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i+1,
                        side=side_osc,
                    ),
                    'out': convention.bodyjoint2name(joint_i),
                    'type': 'STRETCH2FREQ',
                    'weight': (-1 if side_osc else 1)*weight_frequency_down,
                })
        if weight_amplitude_up:
            for joint_i, side_osc in product(
                    range(convention.n_joints_body-1),
                    range(2),
            ):
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_osc,
                    ),
                    'out': convention.bodyjoint2name(joint_i+1),
                    'type': 'STRETCH2AMP',
                    'weight': (-1 if side_osc else 1)*weight_amplitude_up,
                })
        if weight_amplitude_same:
            for joint_i, side_osc in product(
                    range(convention.n_joints_body),
                    range(2),
            ):
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_osc,
                    ),
                    'out': convention.bodyjoint2name(joint_i),
                    'type': 'STRETCH2AMP',
                    'weight': (-1 if side_osc else 1)*weight_amplitude_same,
                })
        if weight_amplitude_down:
            for joint_i, side_osc in product(
                    range(convention.n_joints_body-1),
                    range(2),
            ):
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i+1,
                        side=side_osc,
                    ),
                    'out': convention.bodyjoint2name(joint_i),
                    'type': 'STRETCH2AMP',
                    'weight': (-1 if side_osc else 1)*weight_amplitude_down,
                })
        return connectivity

    @staticmethod
    def default_contact2osc(
            convention,
            w_b_f_up,
            w_b_f_down,
            w_intralimb,
            w_opposite,
            w_following,
            w_diagonal
    ):
        """Default contact sensors to oscillators connectivity"""
        connectivity = []
        n_joints_body = convention.n_joints_body
        # Body
        if w_b_f_up:
            for joint_i, side_o in product(range(n_joints_body), range(2)):
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_o,
                    ),
                    'out': (convention.bodylink2name(link_i=joint_i+1), ''),
                    'type': 'REACTION2FREQ',
                    'weight': w_b_f_up,
                })
        if w_b_f_down:
            for joint_i, side_o in product(range(n_joints_body), range(2)):
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_o,
                    ),
                    'out': (convention.bodylink2name(link_i=joint_i), ''),
                    'type': 'REACTION2FREQ',
                    'weight': w_b_f_down,
                })
        # Intralimb
        iterator = product(
            range(convention.n_legs_pair()),
            range(2),
            range(convention.n_dof_legs),
            range(2),
        )
        if w_intralimb:
            for sensor_leg_i, sensor_side_i, joint_i, side_o in iterator:
                connectivity.append({
                    'in': convention.legosc2name(
                        leg_i=sensor_leg_i,
                        side_i=sensor_side_i,
                        joint_i=joint_i,
                        side=side_o
                    ),
                    'out': (convention.contactleglink2name(
                        leg_i=sensor_leg_i,
                        side_i=sensor_side_i
                    ), ''),
                    'type': 'REACTION2FREQ',
                    'weight': w_intralimb,
                })
        if w_opposite:
            for sensor_leg_i, sensor_side_i, joint_i, side_o in iterator:
                connectivity.append({
                    'in': convention.legosc2name(
                        leg_i=sensor_leg_i,
                        side_i=(sensor_side_i+1) % 2,
                        joint_i=joint_i,
                        side=side_o
                    ),
                    'out': (convention.contactleglink2name(
                        leg_i=sensor_leg_i,
                        side_i=sensor_side_i
                    ), ''),
                    'type': 'REACTION2FREQ',
                    'weight': w_opposite,
                })
        if w_following:
            for sensor_leg_i, sensor_side_i, joint_i, side_o in iterator:
                if sensor_leg_i > 0:
                    connectivity.append({
                        'in': convention.legosc2name(
                            leg_i=sensor_leg_i-1,
                            side_i=sensor_side_i,
                            joint_i=joint_i,
                            side=side_o
                        ),
                        'out': (convention.contactleglink2name(
                            leg_i=sensor_leg_i,
                            side_i=sensor_side_i
                        ), ''),
                        'type': 'REACTION2FREQ',
                        'weight': w_following,
                    })
                if sensor_leg_i < (convention.n_legs_pair() - 1):
                    connectivity.append({
                        'in': convention.legosc2name(
                            leg_i=sensor_leg_i+1,
                            side_i=sensor_side_i,
                            joint_i=joint_i,
                            side=side_o
                        ),
                        'out': (convention.contactleglink2name(
                            leg_i=sensor_leg_i,
                            side_i=sensor_side_i
                        ), ''),
                        'type': 'REACTION2FREQ',
                        'weight': w_following,
                    })
        if w_diagonal:
            for sensor_leg_i, sensor_side_i, joint_i, side_o in iterator:
                if sensor_leg_i > 0:
                    connectivity.append({
                        'in': convention.legosc2name(
                            leg_i=sensor_leg_i-1,
                            side_i=(sensor_side_i+1) % 2,
                            joint_i=joint_i,
                            side=side_o
                        ),
                        'out': (convention.contactleglink2name(
                            leg_i=sensor_leg_i,
                            side_i=sensor_side_i
                        ), ''),
                        'type': 'REACTION2FREQ',
                        'weight': w_diagonal,
                    })
                if sensor_leg_i < (convention.n_legs_pair() - 1):
                    connectivity.append({
                        'in': convention.legosc2name(
                            leg_i=sensor_leg_i+1,
                            side_i=(sensor_side_i+1) % 2,
                            joint_i=joint_i,
                            side=side_o
                        ),
                        'out': (convention.contactleglink2name(
                            leg_i=sensor_leg_i,
                            side_i=sensor_side_i
                        ), ''),
                        'type': 'REACTION2FREQ',
                        'weight': w_diagonal,
                    })
        return connectivity

    @staticmethod
    def default_xfrc2osc(
            convention,
            weight_frequency_up,
            weight_frequency_down,
            weight_amplitude_up,
            weight_amplitude_down,
    ):
        """Default xfrc sensors to oscillators connectivity"""
        connectivity = []
        iterator = product(range(convention.n_joints_body), range(2))
        if weight_frequency_up:
            for joint_i, side_osc in iterator:
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_osc
                    ),
                    'out': convention.bodylink2name(joint_i+1),
                    'type': 'LATERAL2FREQ',
                    'weight': weight_frequency_up,
                })
        if weight_frequency_down:
            for joint_i, side_osc in iterator:
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_osc
                    ),
                    'out': convention.bodylink2name(joint_i),
                    'type': 'LATERAL2FREQ',
                    'weight': weight_frequency_down,
                })
        if weight_amplitude_up:
            for joint_i, side_osc in iterator:
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_osc
                    ),
                    'out': convention.bodylink2name(joint_i+1),
                    'type': 'LATERAL2AMP',
                    'weight': weight_amplitude_up,
                })
        if weight_amplitude_down:
            for joint_i, side_osc in iterator:
                connectivity.append({
                    'in': convention.bodyosc2name(
                        joint_i=joint_i,
                        side=side_osc
                    ),
                    'out': convention.bodylink2name(joint_i),
                    'type': 'LATERAL2AMP',
                    'weight': weight_amplitude_down,
                })
        return connectivity


class AmphibiousOscillatorOptions(Options):
    """Amphibious oscillator options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name = kwargs.pop('name')
        self.initial_phase = kwargs.pop('initial_phase')
        self.initial_amplitude = kwargs.pop('initial_amplitude')
        self.frequency_gain = kwargs.pop('frequency_gain')
        self.frequency_bias = kwargs.pop('frequency_bias')
        self.frequency_low = kwargs.pop('frequency_low')
        self.frequency_high = kwargs.pop('frequency_high')
        self.frequency_saturation_low = kwargs.pop('frequency_saturation_low')
        self.frequency_saturation_high = kwargs.pop('frequency_saturation_high')
        self.amplitude_gain = kwargs.pop('amplitude_gain')
        self.amplitude_bias = kwargs.pop('amplitude_bias')
        self.amplitude_low = kwargs.pop('amplitude_low')
        self.amplitude_high = kwargs.pop('amplitude_high')
        self.amplitude_saturation_low = kwargs.pop('amplitude_saturation_low')
        self.amplitude_saturation_high = kwargs.pop('amplitude_saturation_high')
        self.rate = kwargs.pop('rate')
        self.modular_phase = kwargs.pop('modular_phase')
        self.modular_amplitude = kwargs.pop('modular_amplitude')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousDriveOptions(Options):
    """Amphibious drive options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop('name')
        self.initial_value: float = kwargs.pop('initial_value')
        self.kind: DriveKind = kwargs.pop('kind')
        self.contacts = kwargs.pop('contacts')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousMuscleSetOptions(Options):
    """Amphibious muscle options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.joint_name: str = kwargs.pop('joint_name')
        self.osc1: str = kwargs.pop('osc1')
        self.osc2: str = kwargs.pop('osc2')
        self.alpha: float = kwargs.pop('alpha')  # Gain
        self.beta: float = kwargs.pop('beta')  # Stiffness gain
        self.gamma: float = kwargs.pop('gamma')  # Tonic gain
        self.delta: float = kwargs.pop('delta')  # Damping coefficient
        self.epsilon: float = kwargs.pop('epsilon')  # Friction coefficient
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousAdhesionsOptions(Options):
    """Amphibious adhesion options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.link_name: str = kwargs.pop('link_name')
        self.force: str = kwargs.pop('force')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousVisualsOptions(Options):
    """Amphibious visual options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.visual_name: str = kwargs.pop('visual_name')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousPassiveJointOptions(Options):
    """Amphibious passive joint options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.is_passive: bool = kwargs.pop('is_passive')
        self.stiffness_coefficient: float = kwargs.pop('stiffness_coefficient')
        self.damping_coefficient: float = kwargs.pop('damping_coefficient')
        self.friction_coefficient: float = kwargs.pop('friction_coefficient')
        assert not kwargs, f'Unknown kwargs: {kwargs}'


class AmphibiousSPHOptions(WaterOptions):
    """Amphibious water options"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SPH
        self.sph: bool = kwargs.pop('sph')
        # Logging frequency
        self.sph_log_freq: float = kwargs.pop('sph_log_freq', None)
        # Particle spacing
        self.sph_spacing: float = kwargs.pop('sph_spacing', None)
        # Particle hdx
        self.sph_hdx: float = kwargs.pop('sph_hdx', None)
        # Solid particle spatial density factor
        self.sph_factor_solid: float = kwargs.pop('sph_factor_solid', None)
        # Solid particle mass density
        self.sph_density_solid: float = kwargs.pop('sph_density_solid', None)
        # Fluid density
        self.sph_rho_fluid: float = kwargs.pop('sph_rho_fluid', None)
        # Fluid alpha
        self.sph_alpha: float = kwargs.pop('sph_alpha', None)
        # Fluid beta
        self.sph_beta: float = kwargs.pop('sph_beta', None)
        # Fluid gamma
        self.sph_gamma: float = kwargs.pop('sph_gamma', None)
        # Fluid speed of sound
        self.sph_co: float = kwargs.pop('sph_co', None)
        # XSPH eps
        self.sph_xsph_eps: float = kwargs.pop('sph_xsph_eps', None)
        # Fluid depth
        self.sph_depth: float = kwargs.pop('sph_depth', None)
        # Rigid body multiplier for h
        self.sph_multiplier_h: float = kwargs.pop('sph_multiplier_h', None)
        # Rigid body multiplier for mass
        self.sph_multiplier_mass: float = kwargs.pop('sph_multiplier_mass', None)
        # Rigid body multiplier for volume
        self.sph_multiplier_volume: float = kwargs.pop('sph_multiplier_volume', None)
        # Rigid body multiplier for rad_s
        self.sph_multiplier_rad_s: float = kwargs.pop('sph_multiplier_rad_s', None)


class AmphibiousArenaOptions(ArenaOptions):
    """Amphibious arena options"""

    def __init__(
            self,
            sdf: str,
            spawn: Union[SpawnOptions, Dict],
            water: Union[WaterOptions, Dict],
            ground_height: float,
    ):
        super().__init__(
            sdf=sdf,
            spawn=spawn,
            water=(
                water
                if isinstance(water, WaterOptions)
                else AmphibiousSPHOptions(**water)
                if any(key in water for key in options_kwargs_sph_keys())
                else WaterOptions(**water)
            ),
            ground_height=ground_height,
        )
