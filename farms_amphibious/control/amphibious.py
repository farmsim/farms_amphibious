"""Amphibious controller"""

import os
from typing import Dict, List, Tuple, Callable, Union

import numpy as np

from farms_core.model.options import ControlOptions
from farms_core.model.control import AnimatController, ControlType

from ..data.data import AmphibiousData
from ..model.options import (
    AmphibiousOptions,
    AmphibiousControlOptions,
    KinematicsControlOptions,
)

from .kinematics import KinematicsController
from .drive import DescendingDrive, drive_from_config
from .network import NetworkODE
from .position_muscle_cy import PositionMuscleCy
from .position_phase_cy import PositionPhaseCy
from .passive_cy import PassiveJointCy
from .ekeberg import EkebergMuscleCy


def get_amphibious_controller(animat_data, animat_options, sim_options):
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
        )
    raise Exception('Unknown control options type: {type(animat_options)}')


class AmphibiousController(AnimatController):
    """Amphibious network"""

    def __init__(
            self,
            joints_names: List[str],
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
            drive: DescendingDrive = None,
    ):
        joints_control_types: Dict[str, List[ControlType]] = {
            motor.joint_name: ControlType.from_string_list(motor.control_types)
            for motor in animat_options.control.motors
        }
        super().__init__(
            joints_names=AnimatController.joints_from_control_types(
                joints_names=joints_names,
                joints_control_types=joints_control_types,
            ),
            max_torques=AnimatController.max_torques_from_control_types(
                joints_names=joints_names,
                max_torques={
                    motor.joint_name: motor.limits_torque[1]
                    for motor in animat_options.control.motors
                },
                joints_control_types=joints_control_types,
            ),
        )
        self.network: NetworkODE = NetworkODE(animat_data)
        self.animat_data: AmphibiousData = animat_data
        self.drive: Union[DescendingDrive, None] = drive

        # joints
        joints_map: JointsMap = JointsMap(
            joints=self.joints_names,
            joints_names=joints_names,
            animat_options=animat_options,
        )

        # Equations
        equations = {
            motor.joint_name: motor.equation
            for motor in animat_options.control.motors
        }
        self.equations: Tuple[List[Callable]] = [[], [], []]

        # Muscles
        muscle_map: MusclesMap = MusclesMap(
            joints=joints_names,
            animat_options=animat_options,
            animat_data=animat_data,
        )

        # Network to joints interface
        self.network2joints = {}

        # Position control
        if 'position' in equations.values():
            self.equations[ControlType.POSITION] += [self.positions_network]
            joints_indices = np.array([
                motor_i
                for motor_i, motor in enumerate(animat_options.control.motors)
                if motor.equation == 'position'
            ], dtype=np.uintc)
            joints_names = np.array(
                self.animat_data.sensors.joints.names,
                dtype=object,
            )[joints_indices].tolist()

            self.network2joints['position'] = PositionMuscleCy(
                joints_names=joints_names,
                joints_data=self.animat_data.sensors.joints,
                indices=joints_indices,
                state=self.animat_data.state,
                parameters=np.array(muscle_map.arrays, dtype=np.double),
                osc_indices=np.array(muscle_map.osc_indices, dtype=np.uintc),
                gain=np.array(joints_map.transform_gain, dtype=np.double),
                bias=np.array(joints_map.transform_bias, dtype=np.double),
            )

        # Phase control
        if 'phase' in equations.values():
            self.equations[ControlType.POSITION] += [self.phases_network]
            joints_indices = np.array([
                motor_i
                for motor_i, motor in enumerate(animat_options.control.motors)
                if motor.equation == 'phase'
            ], dtype=np.uintc)
            joints_names = np.array(
                self.animat_data.sensors.joints.names,
                dtype=object,
            )[joints_indices].tolist()
            self.network2joints['phase'] = PositionPhaseCy(
                joints_names=joints_names,
                joints_data=self.animat_data.sensors.joints,
                indices=joints_indices,
                state=self.animat_data.state,
                osc_indices=np.array(muscle_map.osc_indices, dtype=np.uintc),
                gain=np.array(joints_map.transform_gain, dtype=np.double),
                bias=np.array(joints_map.transform_bias, dtype=np.double),
                weight=-1e6,
                offset=0.25*np.pi,
                threshold=1e-2,
            )

        # Ekeberg muscle model control
        for torque_equation in ['ekeberg_muscle', 'ekeberg_muscle_explicit']:

            if torque_equation not in equations.values():
                continue

            joints_indices = np.array([
                motor_i
                for motor_i, motor in enumerate(animat_options.control.motors)
                if motor.equation == torque_equation
            ], dtype=np.uintc)
            joints_names = np.array(
                self.animat_data.sensors.joints.names,
                dtype=object,
            )[joints_indices].tolist()

            self.equations[ControlType.TORQUE] += [{
                'ekeberg_muscle': self.ekeberg_muscle,
                'ekeberg_muscle_explicit': self.ekeberg_muscle_explicit,
            }[torque_equation]]

            if torque_equation == 'ekeberg_muscle':
                # Velocity (damper)
                self.equations[ControlType.VELOCITY] += [
                    self.velocities_ekeberg_damper,
                ]
                self.velocity_indices_ekeberg = np.array([
                    joint_i
                    for joint_i, joint
                    in enumerate(self.joints_names[ControlType.VELOCITY])
                    if joint in equations
                    and equations[joint] == 'ekeberg_muscle'
                ], dtype=np.uintc)
                self.velocity_targets_ekeberg = np.zeros_like(
                    self.velocity_indices_ekeberg,
                    dtype=np.double,
                )

            self.network2joints[torque_equation] = EkebergMuscleCy(
                joints_names=joints_names,
                joints_data=self.animat_data.sensors.joints,
                indices=joints_indices,
                state=self.animat_data.state,
                parameters=np.array(muscle_map.arrays, dtype=np.double),
                osc_indices=np.array(muscle_map.osc_indices, dtype=np.uintc),
                gain=np.array(joints_map.transform_gain, dtype=np.double),
                bias=np.array(joints_map.transform_bias, dtype=np.double),
            )

        # Passive joint control
        if 'passive' in equations.values():

            joints_indices = np.array([
                motor_i
                for motor_i, motor in enumerate(animat_options.control.motors)
                if motor.equation == 'passive'
            ], dtype=np.uintc)
            joints_names = np.array(
                self.animat_data.sensors.joints.names,
                dtype=object,
            )[joints_indices].tolist()

            self.equations[ControlType.TORQUE] += [self.passive]
            self.equations[ControlType.VELOCITY] += [
                self.velocities_passive_damper,
            ]
            self.velocity_indices_passive = np.array([
                joint_i
                for joint_i, joint
                in enumerate(self.joints_names[ControlType.VELOCITY])
                if joint in equations
                and equations[joint] == 'passive'
            ], dtype=np.uintc)
            self.velocity_targets_passive = np.zeros_like(
                self.velocity_indices_passive,
                dtype=np.double,
            )

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
                joints_names=joints_names,
                joints_data=self.animat_data.sensors.joints,
                indices=joints_indices,
                gain=np.array(joints_map.transform_gain, dtype=np.double),
                bias=np.array(joints_map.transform_bias, dtype=np.double),
            )

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ):
        """Control step"""
        if self.drive is not None:
            self.drive.step(iteration, time, timestep)
        self.network.step(iteration, time, timestep)
        for net2joints in self.network2joints.values():
            net2joints.step(iteration)

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
    ) -> Union[Dict[str, float], Tuple]:
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

    def velocities_ekeberg_damper(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Position control to simulate damper properties in Ekeberg muscle"""
        self.max_torques[ControlType.VELOCITY][self.velocity_indices_ekeberg] = np.abs(
            self.network2joints['ekeberg_muscle'].damping(iteration)
            + self.network2joints['ekeberg_muscle'].friction(iteration)
        )
        return dict(zip(
            self.network2joints['ekeberg_muscle'].joints_names,
            self.velocity_targets_ekeberg,
        ))

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

    def velocities_passive_damper(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> Dict[str, float]:
        """Position control to simulate damper properties in Passive joint"""
        self.max_torques[ControlType.VELOCITY][self.velocity_indices_passive] = np.abs(
            self.network2joints['passive'].damping(iteration)
            + self.network2joints['passive'].friction(iteration)
        )
        return dict(zip(
            self.network2joints['passive'].joints_names,
            self.velocity_targets_passive,
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


class JointsMap:
    """Joints map"""

    def __init__(
            self,
            joints: Tuple[List[str]],
            joints_names: List[str],
            animat_options: AmphibiousOptions,
    ):
        super().__init__()
        control_types = list(ControlType)
        self.names = np.array(joints_names)
        self.indices = [  # Indices in animat data for specific control type
            np.array([
                joint_i
                for joint_i, joint in enumerate(joints_names)
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
            for joint in joints_names
        ])
        transform_bias = {
            motor.joint_name: motor.transform.bias
            for motor in animat_options.control.motors
        }
        self.transform_bias = np.array([
            transform_bias[joint]
            for joint in joints_names
        ])


class MusclesMap:
    """Muscles map"""

    def __init__(
            self,
            joints: Tuple[List[str]],
            animat_options: AmphibiousOptions,
            animat_data: AmphibiousData,
    ):
        super().__init__()
        joint_muscle_map = {
            muscle.joint_name: muscle
            for muscle in animat_options.control.muscles
        }
        muscles = [
            joint_muscle_map[joint]
            if joint in joint_muscle_map
            else None
            for joint in joints
        ]
        self.arrays = np.array([
            [
                muscle.alpha, muscle.beta,
                muscle.gamma, muscle.delta,
                muscle.epsilon,
            ]
            if muscle is not None
            else [np.finfo(np.double).max]*5
            for muscle in muscles
        ], dtype=np.double)
        osc_names = animat_data.network.oscillators.names
        self.osc_indices = np.array([
            [
                osc_names.index(muscle.osc1)
                if muscle is not None
                else np.iinfo(np.uintc).max
                for muscle in muscles
            ],
            [
                osc_names.index(muscle.osc2)
                if muscle is not None and muscle.osc2 is not None
                else np.iinfo(np.uintc).max
                for muscle in muscles
             ],
        ], dtype=np.uintc)
