"""Amphibious controller

The controller class must implement three methods:

    - positions(...)

    - velocities(...)

    - torques(...)

On the other hand, joints can be based on three different types. Either the
joint is position controlled, muscle controlled with Ekeberg's model or
passively controlled. Previously, either all joints were position or torque
contolled. Recently, the new Ekeberg muscle implementation required using both
velocity and torque control for all joints. However, the new requirements mean
that we could have a mix between joints being position controlled, torque and
velocity controller or torque-only controlled. To accomodate for this new
requirement, Cython could be used to facilitate the implementation. This would
allow to use for-loops without worrying about the computational cost of looping
in Python and run a separate equation for each joint independently. Each of the
methods could then iterate through the joints and query the equation to be used
in order to return the final output. It would also allow to share more code
between the different Ekeberg muscle implementations.

"""

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from dm_control.rl.control import Task
from dm_control.mjcf.physics import Physics

from farms_core.io.yaml import yaml2pyobject
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.control import AnimatController, ControlType
from farms_core.extensions.extensions import import_item
from farms_core.sensors.sensor_convention import sc

from ..data.data import AmphibiousData
from ..model.options import AmphibiousOptions

from .drive import DescendingDrive
from .network import AnimatNetwork, NetworkODE
from .position_muscle_cy import PositionMuscleCy
from .position_phase_cy import PositionPhaseCy
from .passive_cy import PassiveJointCy
from .ekeberg import EkebergMuscleCy


class UnknownController(Exception):
    """Unknown controller"""


class JointMuscleController(AnimatController):
    """Joint muscle controller"""

    def __init__(
            self,
            animat_i: int,
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
            animat_network: AnimatNetwork | None,
    ):
        joints_control_names = animat_options.control.joints_names()
        joints_control_types: dict[str, list[ControlType]] = {
            motor.joint_name: ControlType.from_string_list(motor.control_types)
            for motor in animat_options.control.motors
        }
        super().__init__(
            animat_i=animat_i,
            joints_names=AnimatController.joints_from_control_types(
                joints_names=joints_control_names,
                joints_control_types=joints_control_types,
            ),
            muscles_names=[],
            max_torques=AnimatController.max_torques_from_control_types(
                joints_names=joints_control_names,
                max_torques={
                    motor.joint_name: motor.limits_torque[1]
                    for motor in animat_options.control.motors
                },
                joints_control_types=joints_control_types,
            ),
            substep=True,
        )

        self.network: AnimatNetwork | None = animat_network
        self.animat_data: AnimatData = animat_data

        # joints
        self.joints_map: JointsMap = JointsMap(
            joints=self.joints_names,
            joints_sensors_names=self.animat_data.sensors.joints.names,
            animat_options=animat_options,
        )

        # Equations
        self.equations_dict = {
            motor.joint_name: motor.equation
            for motor in animat_options.control.motors
        }
        self.equations: tuple[list[Callable]] = [[], [], []]

        # Muscles
        self.muscle_maps: dict[str, MusclesMap] = {}

        # Network to joints interface
        self.network2joints = {}

        # Ekeberg muscle model control
        for torque_equation in ['ekeberg_muscle', 'ekeberg_muscle_explicit']:

            if torque_equation not in self.equations_dict.values():
                continue

            muscles_joints: list[str] = [
                motor.joint_name
                for motor in animat_options.control.motors
                if motor.equation == torque_equation
            ]
            muscles_joints_indices = np.array([
                self.animat_data.sensors.joints.names.index(joint_name)
                for joint_name in muscles_joints
            ], dtype=np.uintc)
            self.muscle_maps[torque_equation] = MusclesMap(
                joints=muscles_joints,
                animat_options=animat_options,
                animat_data=animat_data,
            )

            self.equations[ControlType.TORQUE] += [{
                'ekeberg_muscle': self.ekeberg_muscle,
                'ekeberg_muscle_explicit': self.ekeberg_muscle_explicit,
            }[torque_equation]]

            muscle_map = self.muscle_maps[torque_equation]
            self.network2joints[torque_equation] = EkebergMuscleCy(
                joints_names=muscles_joints,
                joints_data=self.animat_data.sensors.joints,
                indices=muscles_joints_indices,
                state=self.animat_data.state,
                parameters=np.array(muscle_map.arrays, dtype=np.double),
                osc_indices=np.array(muscle_map.osc_indices, dtype=np.uintc),
                gain=np.array(self.joints_map.transform_gain, dtype=np.double),
                bias=np.array(self.joints_map.transform_bias, dtype=np.double),
            )

        # Passive joint control
        if 'passive' in self.equations_dict.values():

            self.equations[ControlType.TORQUE] += [self.passive]
            passive_joints: list[str] = [
                motor.joint_name
                for motor in animat_options.control.motors
                if motor.equation == 'passive'
            ]
            passive_joints_indices = np.array([
                self.animat_data.sensors.joints.names.index(joint_name)
                for joint_name in passive_joints
            ], dtype=np.uintc)
            self.network2joints['passive'] = PassiveJointCy(
                stiffness_coefficients=np.array([
                    motor.passive.stiffness_coefficient
                    for motor in animat_options.control.motors
                    if motor.equation == 'passive'
                ], dtype=np.double),
                damping_coefficients=np.array([
                    motor.passive.damping_coefficient
                    for motor in animat_options.control.motors
                    if motor.equation == 'passive'
                ], dtype=np.double),
                friction_coefficients=np.array([
                    motor.passive.friction_coefficient
                    for motor in animat_options.control.motors
                    if motor.equation == 'passive'
                ], dtype=np.double),
                joints_names=passive_joints,
                joints_data=self.animat_data.sensors.joints,
                indices=passive_joints_indices,
                gain=np.array(self.joints_map.transform_gain, dtype=np.double),
                bias=np.array(self.joints_map.transform_bias, dtype=np.double),
            )

    def before_step(self, task: Task, action, physics: Physics):
        """Before step"""
        del action
        index = task.iteration % task.buffer_size
        self.network.step(
            index=index,
            time=physics.time()/task.units.seconds,
            timestep=physics.timestep()/task.units.seconds,
        )
        for net2joints in self.network2joints.values():
            net2joints.step(index)

    def positions(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Positions"""
        output = {}
        for equation in self.equations[ControlType.POSITION]:
            output.update(equation(iteration, time, timestep))
        return output

    def velocities(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Velocities"""
        output: dict[str, float] = {}
        for equation in self.equations[ControlType.VELOCITY]:
            output.update(equation(iteration, time, timestep))
        return output

    def torques(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Torques"""
        output = {}
        for equation in self.equations[ControlType.TORQUE]:
            output.update(equation(iteration, time, timestep))
        return output

    def springrefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Spring references"""
        output = {}
        if 'ekeberg_muscle' in self.network2joints:
            output = dict(zip(
                self.network2joints['ekeberg_muscle'].joints_names,
                self.network2joints['ekeberg_muscle'].joints_offsets,
            ))
        return output

    def springcoefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Spring coefficients"""
        output = {}
        if 'ekeberg_muscle' in self.network2joints:
            output = dict(zip(
                self.network2joints['ekeberg_muscle'].joints_names,
                self.network2joints['ekeberg_muscle'].spring_coefs,
            ))
        return output

    def dampingcoefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Damping coefficients"""
        output = {}
        if 'ekeberg_muscle' in self.network2joints:
            output = dict(zip(
                self.network2joints['ekeberg_muscle'].joints_names,
                self.network2joints['ekeberg_muscle'].damping_coefs,
            ))
        return output

    def ekeberg_muscle(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Ekeberg muscle"""
        return dict(zip(
            self.network2joints['ekeberg_muscle'].joints_names,
            self.network2joints['ekeberg_muscle'].torques_implicit(iteration),
        ))

    def ekeberg_muscle_spring_ref(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Ekeberg muscle spring reference"""
        return dict(zip(
            self.network2joints['ekeberg_muscle'].joints_names,
            self.network2joints['ekeberg_muscle'].springrefs(iteration),
        ))

    def ekeberg_muscle_explicit(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Ekeberg muscle with explicit passive dynamics"""
        key = 'ekeberg_muscle_explicit'
        return dict(zip(
            self.network2joints[key].joints_names,
            self.network2joints[key].torque_cmds(iteration),
        ))

    def passive(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Passive joint"""
        return dict(zip(
            self.network2joints['passive'].joints_names,
            self.network2joints['passive'].stiffness(iteration),
        ))

    def passive_explicit(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Passive joint with explicit passive dynamics"""
        return dict(zip(
            self.network2joints['passive'].joints_names,
            self.network2joints['passive'].torque_cmds(iteration),
        ))


class AmphibiousController(JointMuscleController):
    """Amphibious network"""

    def __init__(
            self,
            animat_i: int,
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
            animat_network: AnimatNetwork | None,
            drive: DescendingDrive | None = None,
    ):
        self.drive = drive
        super().__init__(
            animat_i=animat_i,
            animat_options=animat_options,
            animat_data=animat_data,
            animat_network=animat_network,
        )

        # Position control
        if 'position_muscle' in self.equations_dict.values():
            self.equations[ControlType.POSITION] += [self.positions_network]
            muscles_joints: list[str] = [
                motor.joint_name
                for motor in animat_options.control.motors
                if motor.equation == 'position_muscle'
            ]
            muscles_joints_indices = np.array([
                self.animat_data.sensors.joints.names.index(joint_name)
                for joint_name in muscles_joints
            ], dtype=np.uintc)
            self.muscle_maps['position_muscle'] = MusclesMap(
                joints=muscles_joints,
                animat_options=animat_options,
                animat_data=animat_data,
            )
            muscle_map = self.muscle_maps['position_muscle']
            self.network2joints['position_muscle'] = PositionMuscleCy(
                joints_names=muscles_joints,
                joints_data=self.animat_data.sensors.joints,
                indices=muscles_joints_indices,
                state=self.animat_data.state,
                parameters=np.array(muscle_map.arrays, dtype=np.double),
                osc_indices=np.array(muscle_map.osc_indices, dtype=np.uintc),
                gain=np.array(self.joints_map.transform_gain, dtype=np.double),
                bias=np.array(self.joints_map.transform_bias, dtype=np.double),
            )

        # Phase control
        if 'position_phase' in self.equations_dict.values():
            self.equations[ControlType.POSITION] += [self.phases_network]
            muscles_joints: list[str] = [
                motor.joint_name
                for motor in animat_options.control.motors
                if motor.equation == 'position_phase'
            ]
            muscles_joints_indices = np.array([
                self.animat_data.sensors.joints.names.index(joint_name)
                for joint_name in muscles_joints
            ], dtype=np.uintc)
            self.muscle_maps['position_phase'] = MusclesMap(
                joints=muscles_joints,
                animat_options=animat_options,
                animat_data=animat_data,
            )
            muscle_map = self.muscle_maps['position_phase']
            self.network2joints['position_phase'] = PositionPhaseCy(
                joints_names=muscles_joints,
                joints_data=self.animat_data.sensors.joints,
                indices=muscles_joints_indices,
                state=self.animat_data.state,
                osc_indices=np.array(muscle_map.osc_indices, dtype=np.uintc),
                gain=np.array(self.joints_map.transform_gain, dtype=np.double),
                bias=np.array(self.joints_map.transform_bias, dtype=np.double),
                threshold=1e-2,
            )

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):
        """From options

        animat_options = experiment_options.animats[animat_i]

        """
        del config
        drive = None
        if animat_data.state is None:
            return cls(
                animat_i=animat_i,
                animat_options=animat_options,
                animat_data=animat_data,
                animat_network=None,
                drive=None,
            )
        animat_network = NetworkODE(
            data=animat_data,
            integrator='dopri5',
            nsteps=1000,
            max_step=experiment_options.simulation.physics.timestep,
            verbosity=3,
        )
        if (
                animat_options.control.network.drive_config
                and 'drive_config' in animat_options.control.network
        ):
            network_options = animat_options.control.network
            filename = network_options.drive_config
            drive_config = yaml2pyobject(filename)
            loader = network_options.drive_loader
            assert network_options.drive_loader, (
                f'Cannot load {filename} without knowing {loader=}'
                f'\nDrive config:\n\n{drive_config}'
            )
            drive_loader = import_item(loader)
            drive = drive_loader.from_options(
                animat_data,
                animat_options,
                drive_config,
                experiment_options.simulation,
            )
        return cls(
            animat_i=animat_i,
            animat_options=animat_options,
            animat_data=animat_data,
            animat_network=animat_network,
            drive=drive,
        )

    def initialize_episode(self, task: Task, physics: Physics):
        """Initialize episode"""
        self.animat_data.sensors.links.array[1:, :, :] = 0
        self.animat_data.sensors.joints.array[1:, :, :] = 0
        self.animat_data.sensors.contacts.array[1:, :, :] = 0
        self.animat_data.sensors.xfrc.array[1:, :, :] = 0
        if self.drive is not None:
            self.drive.drives.array[1:, :] = 0
        self.network.initialize_episode()

    def before_step(self, task: Task, action, physics: Physics):
        """Before step"""
        del action
        time = physics.time()/task.units.seconds
        timestep = physics.timestep()/task.units.seconds
        index = task.iteration % task.buffer_size
        self.step(iteration=index, time=time, timestep=timestep)

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ):
        """Control step

        This function is needed for running the controller without simulation.

        """
        if self.drive is not None:
            self.drive.step(iteration, time, timestep)
        if self.network is not None:
            self.network.step(iteration, time, timestep)
        for net2joints in self.network2joints.values():
            if net2joints is not None:
                net2joints.step(iteration)

    def positions_network(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Positions network"""
        return dict(zip(
            self.network2joints['position_muscle'].joints_names,
            self.network2joints['position_muscle'].position_cmds(iteration),
        ))

    def phases_network(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Phases network"""
        return dict(zip(
            self.network2joints['position_phase'].joints_names,
            self.network2joints['position_phase'].position_cmds(iteration),
        ))


class AmphibiousDriveController(AmphibiousController):
    """Amphibious network"""

    def __init__(
            self,
            animat_i: int,
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
            animat_network: AnimatNetwork,
            drive: DescendingDrive,  # | None
    ):
        # self.drive: DescendingDrive = drive  # | None
        self.cmap_drives = plt.get_cmap('turbo')
        self.cmap_phases = plt.get_cmap('Greens')
        self.norm = mcolors.Normalize(vmin=0, vmax=6)
        super().__init__(
            animat_i=animat_i,
            animat_options=animat_options,
            animat_data=animat_data,
            animat_network=animat_network,
            drive=drive,
        )
        self.visuals: VisualsArray = self.animat_data.sensors.visuals

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ):
        """Control step"""
        if self.drive is not None:
            self.drive.step(iteration, time, timestep)
        super().step(iteration, time, timestep)
        if self.visuals.shape[1] > 0 and hasattr(self.drive, 'drives'):
            self.set_visuals(iteration)
        else:
            self.set_visuals_invisible(iteration)

    def set_visuals(self, iteration: int):
        """Set visuals"""
        values = np.array(self.drive.drives.array[iteration, :])
        colors = self.cmap_drives(self.norm(values))
        values = np.array(self.drive.drives.array[iteration, :])
        emissions = self.cmap_phases(self.norm(values))
        phases = self.animat_data.state.phases(iteration)
        amplitudes = self.animat_data.state.amplitudes(iteration)
        oscillation_amplitude = 2.0
        outputs = 0.5*oscillation_amplitude*(
            1+np.cos(np.where(np.array(amplitudes) > 1e-3, phases, np.pi))
        )
        color_start, color_end = sc.visual_color_r, sc.visual_color_a+1
        emission_start, emission_end = sc.visual_emission_r, sc.visual_emission_i+1
        self.visuals.array[iteration, :, color_start:color_end] = colors
        self.visuals.array[iteration, :, emission_start:emission_end] = emissions
        self.visuals.array[iteration, :2, sc.visual_emission_i] = np.zeros(2)
        self.visuals.array[iteration, 2:, sc.visual_emission_i] = outputs

    def set_visuals_invisible(self, iteration: int):
        """Set visuals"""
        self.visuals.array[iteration, :, :] = 0


class JointsMap:
    """Joints map"""

    def __init__(
            self,
            joints: tuple[list[str]],
            joints_sensors_names: list[str],
            animat_options: AmphibiousOptions,
    ):
        super().__init__()
        control_types = list(ControlType)
        self.indices = [  # Indices in animat data for specific control type
            np.array([
                joint_i
                for joint_i, joint in enumerate(joints_sensors_names)
                if joint in joints[control_type]
            ])
            for control_type in control_types
        ]
        transform_gains = {
            motor.joint_name: motor.transform.gain
            for motor in animat_options.control.motors
        }
        self.transform_gain = np.array([
            transform_gains[joint]
            for joint in joints_sensors_names
        ])
        transform_bias = {
            motor.joint_name: motor.transform.bias
            for motor in animat_options.control.motors
        }
        self.transform_bias = np.array([
            transform_bias[joint]
            for joint in joints_sensors_names
        ])


class MusclesMap:
    """Muscles map

    For amphibious control, we have oscillators, muscles and joints. This muscle
    map allows us to easily obtain the muscle parameters and oscillators for
    each joint.

    """

    def __init__(
            self,
            joints: list[str],
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
    ):
        super().__init__()
        joint_muscle_map = {
            muscle.joint_name: muscle
            for muscle in animat_options.control.muscles
        }
        for joint in joints:
            assert joint in joint_muscle_map, (
                f"{joint=} not in {joint_muscle_map.keys()=}"
            )
        muscles = [
            joint_muscle_map[joint]
            for joint in joints
        ]
        self.arrays = np.array([
            [
                muscle.alpha, muscle.beta,
                muscle.gamma, muscle.delta,
                muscle.epsilon,
            ]
            for muscle in muscles
        ], dtype=np.double)
        if animat_data.network is None:
            self.osc_indices = np.array([[],[]], dtype=np.uintc)
            return
        osc_names = animat_data.network.oscillators.names
        self.osc_indices = np.array([
            [
                osc_names.index(muscle.osc1)
                if muscle.osc1 in osc_names
                else np.iinfo(np.uintc).max
                for muscle in muscles
            ],
            [
                osc_names.index(muscle.osc2)
                if muscle.osc2 in osc_names
                else np.iinfo(np.uintc).max
                for muscle in muscles
            ],
        ], dtype=np.uintc)
