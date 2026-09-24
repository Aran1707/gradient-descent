"""Adaptive and sign-based optimizers used by the seminar."""

from __future__ import annotations

import numpy as np

from .base import Optimizer


class AdaGrad(Optimizer):
    def __init__(self, lr: float, eps: float = 1e-8):
        super().__init__(lr)
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.accumulator: dict[str, np.ndarray] = {}

    def step(self, params) -> None:
        for parameter, gradient, key in self._materialize(params):
            accumulator = self._state_array(self.accumulator, key, parameter)
            accumulator[...] += gradient * gradient
            parameter[...] -= self.lr * gradient / (np.sqrt(accumulator) + self.eps)


class RMSProp(Optimizer):
    def __init__(self, lr: float, rho: float = 0.9, eps: float = 1e-8):
        super().__init__(lr)
        if not 0 <= rho < 1:
            raise ValueError("rho must be in the range [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.rho = float(rho)
        self.eps = float(eps)
        self.average: dict[str, np.ndarray] = {}

    def step(self, params) -> None:
        for parameter, gradient, key in self._materialize(params):
            average = self._state_array(self.average, key, parameter)
            average[...] = self.rho * average + (1 - self.rho) * (gradient * gradient)
            parameter[...] -= self.lr * gradient / (np.sqrt(average) + self.eps)


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(lr)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 and beta2 must be in the range [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.t = 0
        self.first_moment: dict[str, np.ndarray] = {}
        self.second_moment: dict[str, np.ndarray] = {}

    def step(self, params) -> None:
        materialized = self._materialize(params)
        self.t += 1
        for parameter, gradient, key in materialized:
            first = self._state_array(self.first_moment, key, parameter)
            second = self._state_array(self.second_moment, key, parameter)
            first[...] = self.beta1 * first + (1 - self.beta1) * gradient
            second[...] = self.beta2 * second + (1 - self.beta2) * (gradient * gradient)
            first_hat = first / (1 - self.beta1**self.t)
            second_hat = second / (1 - self.beta2**self.t)
            parameter[...] -= self.lr * first_hat / (np.sqrt(second_hat) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay."""

    def __init__(self, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs):
        super().__init__(lr=lr, **kwargs)
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.weight_decay = float(weight_decay)

    def step(self, params) -> None:
        materialized = self._materialize(params)
        if self.weight_decay:
            decay = 1.0 - self.lr * self.weight_decay
            for parameter, _, _ in materialized:
                parameter[...] *= decay
        super().step(materialized)


class Lion(Optimizer):
    """Sign-based momentum optimizer for an advanced seminar comparison."""

    def __init__(
        self,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
    ):
        super().__init__(lr)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 and beta2 must be in the range [0, 1)")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.momentum: dict[str, np.ndarray] = {}

    def step(self, params) -> None:
        for parameter, gradient, key in self._materialize(params):
            momentum = self._state_array(self.momentum, key, parameter)
            update = self.beta1 * momentum + (1 - self.beta1) * gradient
            parameter[...] -= self.lr * np.sign(update)
            momentum[...] = self.beta2 * momentum + (1 - self.beta2) * gradient
