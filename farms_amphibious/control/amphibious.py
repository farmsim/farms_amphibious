"""Amphibious controller"""

import os
from typing import Dict, List, Tuple, Callable, Union

import numpy as np
from dm_control.rl.control import Task
from dm_control.mjcf.physics import Physics

from farms_core.io.yaml import yaml2pyobject
from farms_core.model.data import AnimatData
from farms_core.model.options import AnimatOptions
from farms_core.experiment.options import ExperimentOptions
from farms_core.model.control import AnimatController, ControlType
from farms_core.simulation.options import SimulationOptions
from farms_core.extensions.extensions import import_item

from ..data.data import AmphibiousData
from ..model.options import (
    AmphibiousOptions,
    AmphibiousControlOptions,
    KinematicsControlOptions,
)

from .kinematics import KinematicsController
from .drive import DescendingDrive, drive_from_config
from .network import AnimatNetwork, NetworkODE
from .position_muscle_cy import PositionMuscleCy
from .position_phase_cy import PositionPhaseCy
from .passive_cy import PassiveJointCy
from .ekeberg import EkebergMuscleCy


def get_amphibious_controller(
        animat_data: AnimatData,
        animat_options: AnimatOptions,
        sim_options: SimulationOptions,
        **kwargs,
):
    """Controller from config"""
    joints_names = animat_options.control.joints_names()
    if isinstance(animat_options.control, AmphibiousControlOptions):
        return AmphibiousController(
            joints_names=joints_names,
            animat_options=animat_options,
            animat_data=animat_data,
            drive=(
                drive_from_config(
                    filename=animat_options.control.network.drive_config,
                    animat_data=animat_data,
                    simulation_options=sim_options,
                )
                if animat_options.control.network is not None
                and animat_options.control.network.drive_config
                and 'drive_config' in animat_options.control.network
                else None
            ),
            **kwargs,
        )
    joints_control_types = {
        motor.joint_name: ControlType.from_string_list(
            motor.control_types,
        )
        for motor in animat_options.control.motors
    }
    joints_names_per_type = AnimatController.joints_from_control_types(
        joints_names=joints_names,
        joints_control_types=joints_control_types,
    )
    max_torques = {
        motor.joint_name: motor.limits_torque[1]
        for motor in animat_options.control.motors
    }
    max_torques_per_type = AnimatController.max_torques_from_control_types(
        joints_names=joints_names,
        max_torques=max_torques,
        joints_control_types=joints_control_types,
    )
    if isinstance(animat_options.control, KinematicsControlOptions):
        assert os.path.isfile(animat_options.control.kinematics_file), (
            f'{animat_options.control.kinematics_file} is not a file'
        )
        return KinematicsController(
            joints_names=joints_names_per_type,
            kinematics=np.genfromtxt(
                animat_options.control.kinematics_file,
                delimiter=',',
            ),
            sampling=animat_options.control.kinematics_sampling,
            indices=animat_options.control.kinematics_indices,
            time_index=animat_options.control.kinematics_time_index,
            invert_motors=animat_options.control.kinematics_invert,
            degrees=animat_options.control.kinematics_degrees,
            timestep=sim_options.timestep,
            n_iterations=sim_options.n_iterations,
            animat_data=animat_data,
            max_torques=max_torques_per_type,
            init_time=animat_options.control.kinematics_start,
            end_time=animat_options.control.kinematics_end,
            **kwargs,
        )
    raise Exception('Unknown control options type: {type(animat_options)}')


class JointMuscleController(AnimatController):
    """Joint muscle controller"""

    def __init__(
            self,
            animat_i: int,
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
            animat_network: AnimatNetwork,
    ):
        joints_control_names = animat_options.control.joints_names()
        joints_control_types: Dict[str, List[ControlType]] = {
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

        self.network: AnimatNetwork = animat_network
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
        self.equations: Tuple[List[Callable]] = [[], [], []]

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
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
        """Velocities"""
        output: Dict[str, float] = {}
        for equation in self.equations[ControlType.VELOCITY]:
            output.update(equation(iteration, time, timestep))
        return output

    def torques(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
        """Ekeberg muscle with explicit passive dynamics"""
        return dict(zip(
            self.network2joints['ekeberg_muscle_explicit'].joints_names,
            self.network2joints['ekeberg_muscle_explicit'].torque_cmds(iteration),
        ))

    def passive(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
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
    ) -> Dict[str, float]:
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
            animat_network: AnimatNetwork,
            drive: DescendingDrive = None,
    ):
        super().__init__(
            animat_i=animat_i,
            animat_options=animat_options,
            animat_data=animat_data,
            animat_network=animat_network,
        )
        self.drive: Union[DescendingDrive, None] = drive

        # Position control
        if 'position' in self.equations_dict.values():
            self.equations[ControlType.POSITION] += [self.positions_network]
            muscles_joints: list[str] = [
                motor.joint_name
                for motor in animat_options.control.motors
                if motor.equation == 'position'
            ]
            muscles_joints_indices = np.array([
                self.animat_data.sensors.joints.names.index(joint_name)
                for joint_name in muscles_joints
            ], dtype=np.uintc)
            self.muscle_maps['position'] = MusclesMap(
                joints=muscles_joints,
                animat_options=animat_options,
                animat_data=animat_data,
            )
            muscle_map = self.muscle_maps['position']
            self.network2joints['position'] = PositionMuscleCy(
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
        if 'phase' in self.equations_dict.values():
            self.equations[ControlType.POSITION] += [self.phases_network]
            muscles_joints: list[str] = [
                motor.joint_name
                for motor in animat_options.control.motors
                if motor.equation == 'phase'
            ]
            muscles_joints_indices = np.array([
                self.animat_data.sensors.joints.names.index(joint_name)
                for joint_name in muscles_joints
            ], dtype=np.uintc)
            self.muscle_maps['phase'] = MusclesMap(
                joints=muscles_joints,
                animat_options=animat_options,
                animat_data=animat_data,
            )
            muscle_map = self.muscle_maps['phase']
            self.network2joints['phase'] = PositionPhaseCy(
                joints_names=muscles_joints,
                joints_data=self.animat_data.sensors.joints,
                indices=muscles_joints_indices,
                state=self.animat_data.state,
                osc_indices=np.array(muscle_map.osc_indices, dtype=np.uintc),
                gain=np.array(self.joints_map.transform_gain, dtype=np.double),
                bias=np.array(self.joints_map.transform_bias, dtype=np.double),
                weight=-1e6,
                offset=0.25*np.pi,
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
        self.network.step(iteration, time, timestep)
        for net2joints in self.network2joints.values():
            net2joints.step(iteration)

    def positions_network(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Positions network"""
        return dict(zip(
            self.network2joints['position'].joints_names,
            self.network2joints['position'].position_cmds(iteration),
        ))

    def phases_network(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Phases network"""
        return dict(zip(
            self.network2joints['phase'].joints_names,
            self.network2joints['phase'].position_cmds(iteration),
        ))


class JointsMap:
    """Joints map"""

    def __init__(
            self,
            joints: Tuple[List[str]],
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
    """Muscles map"""

    def __init__(
            self,
            joints: List[str],
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
        osc_names = animat_data.network.oscillators.names
        self.osc_indices = np.array([
            [
                osc_names.index(muscle.osc1)
                for muscle in muscles
            ],
            [
                osc_names.index(muscle.osc2)
                for muscle in muscles
            ],
        ], dtype=np.uintc)
