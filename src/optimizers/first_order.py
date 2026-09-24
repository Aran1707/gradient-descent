"""First-order optimizers used by the seminar CNN and landscape visualizations."""

from __future__ import annotations

import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """Vanilla gradient descent."""

    def step(self, params) -> None:
        for parameter, gradient, _ in self._materialize(params):
            parameter[...] -= self.lr * gradient


class Momentum(Optimizer):
    """Gradient descent with a velocity-like moving accumulator."""

    def __init__(self, lr: float, beta: float = 0.9):
        super().__init__(lr)
        if not 0 <= beta < 1:
            raise ValueError("beta must be in the range [0, 1)")
        self.beta = float(beta)
        self.velocity: dict[str, np.ndarray] = {}

    def step(self, params) -> None:
        for parameter, gradient, key in self._materialize(params):
            velocity = self._state_array(self.velocity, key, parameter)
            velocity[...] = self.beta * velocity + gradient
            parameter[...] -= self.lr * velocity


class Nesterov(Optimizer):
    """Nesterov-style momentum using the current gradient.

    A fully exact look-ahead implementation would require another model
    forward/backward pass. This reusable parameter-only interface uses the
    common first-order Nesterov update with the current gradient and previous
    velocity, keeping one gradient evaluation per training step.
    """

    def __init__(self, lr: float, beta: float = 0.9):
        super().__init__(lr)
        if not 0 <= beta < 1:
            raise ValueError("beta must be in the range [0, 1)")
        self.beta = float(beta)
        self.velocity: dict[str, np.ndarray] = {}

    def step(self, params) -> None:
        for parameter, gradient, key in self._materialize(params):
            velocity = self._state_array(self.velocity, key, parameter)
            previous_velocity = velocity.copy()
            velocity[...] = self.beta * velocity + gradient
            parameter[...] -= self.lr * (self.beta * previous_velocity + gradient)
