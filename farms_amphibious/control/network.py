"""Network"""

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy import integrate
from scipy.integrate._ode import ode as ODE

from farms_core.model.data import AnimatData

from .ode import ode_oscillators_sparse


class AnimatNetwork(ABC):
    """Animat network"""

    def __init__(self, data, n_iterations):
        super().__init__()
        self.data: AnimatData = data
        self.n_iterations = n_iterations

    @abstractmethod
    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
            **kwargs,
    ):
        """Step function called at each simulation iteration"""


class NetworkODE(AnimatNetwork):
    """NetworkODE"""

    def __init__(self, data, integrator='dopri5', **kwargs):
        state_array = data.state.array
        self.modulo: int = kwargs.pop('modulo', 1)
        super().__init__(data=data, n_iterations=np.shape(state_array)[0])
        self.dstate = np.zeros_like(data.state.array[0, :])
        self.ode: Callable = kwargs.pop('ode', ode_oscillators_sparse)
        self.solver: ODE = integrate.ode(f=self.ode)
        self.solver.set_integrator(integrator, **kwargs)
        self.solver.set_initial_value(y=state_array[0, :], t=0.0)

    def copy_next_drive(self, iteration):
        """Set initial drive"""
        array = self.data.network.drives.array
        array[iteration+1] = array[iteration]

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
            checks: bool = False,
            strict: bool = False,
    ):
        """Control step

        Called after obtaining all the current sensor data, and right before
        calling the physics.

        """
        if iteration == 0:
            self.copy_next_drive(iteration)
            return
        if checks:
            assert np.array_equal(
                self.solver.y,
                np.array(self.data.state.array[iteration-1, :]),
            ), (
                f'At {iteration=}:\n'
                f'{self.solver.y=}'
                ' != '
                f'{np.array(self.data.state.array[iteration-1, :])=}'
            )
        if iteration % self.modulo:
            self.data.state.array[iteration, :] = (
                self.data.state.array[iteration-1, :]
            )
        else:
            self.solver.set_f_params(self.dstate, iteration, self.data)
            while self.solver.successful() and self.solver.t < time+0.99*timestep:
                self.data.state.array[iteration, :] = (
                    self.solver.integrate(time+timestep)  # , step=True, relax=True
                )
            if not self.solver.successful():
                message = (
                    f'ODE not integrated properly at {iteration=}'
                    f' ({self.solver.t=} < {time+timestep=} [s])'
                    f'\nReturn code: {self.solver.get_return_code()=}'
                    f'\nState:\n{np.array(self.data.state.array[iteration, :])}'
                )
                if strict:
                    raise IntegrationException(message)
                pylog.warning('%s\n\nResetting to previous iteration', message)
                self.solver.set_initial_value(y=self.solver.y, t=time+timestep)
        # Handle drive
        self.data.sensors.visuals
        if iteration < self.n_iterations-1:
            self.copy_next_drive(iteration)
        if checks:
            assert self.solver.successful(), (
                f'Solver was not successful at {iteration=}'
            )
            assert abs(time+timestep-self.solver.t) < 1e-6*timestep, (
                'ODE solver time: '
                f'{self.solver.t} [s] != Simulation time: {time+timestep} [s]'
            )
