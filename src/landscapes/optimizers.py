"""Small, readable optimizer trajectory runner for two-dimensional landscapes.

Implements educational 2D versions of the core optimizer family used in the seminar:
SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW, and Lion.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

try:
    from .objectives import grad_ill_conditioned
except ImportError:  # Supports running this file directly as a teaching script.
    from objectives import (
        grad_ill_conditioned,
    )


class SGD:
    """Vanilla gradient descent in 2D."""

    def __init__(self, lr: float):
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = float(lr)

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return theta - self.lr * grad


class Momentum:
    """Gradient descent with classical Polyak momentum accumulator."""

    def __init__(self, lr: float, beta: float = 0.9):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        self.lr = float(lr)
        self.beta = float(beta)
        self.v: np.ndarray | None = None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.beta * self.v + grad
        return theta - self.lr * self.v


class Nesterov:
    """Nesterov accelerated gradient descent (first-order formulation)."""

    def __init__(self, lr: float, beta: float = 0.9):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        self.lr = float(lr)
        self.beta = float(beta)
        self.v: np.ndarray | None = None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(theta)
        prev_v = self.v.copy()
        self.v = self.beta * self.v + grad
        return theta - self.lr * (self.beta * prev_v + grad)


class AdaGrad:
    """AdaGrad optimizer with cumulative sum of squared gradients."""

    def __init__(self, lr: float, eps: float = 1e-8):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.lr = float(lr)
        self.eps = float(eps)
        self.G: np.ndarray | None = None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.G is None:
            self.G = np.zeros_like(theta)
        self.G += grad * grad
        return theta - self.lr * grad / (np.sqrt(self.G) + self.eps)


class RMSProp:
    """RMSProp optimizer with exponential moving average of squared gradients."""

    def __init__(self, lr: float, rho: float = 0.9, eps: float = 1e-8):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= rho < 1:
            raise ValueError("rho must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.lr = float(lr)
        self.rho = float(rho)
        self.eps = float(eps)
        self.v: np.ndarray | None = None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.rho * self.v + (1.0 - self.rho) * (grad * grad)
        return theta - self.lr * grad / (np.sqrt(self.v) + self.eps)


class Adam:
    """Adam optimizer with first and second moment moving averages and bias correction."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1**self.t)
        v_hat = self.v / (1.0 - self.beta2**self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay."""

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        self.weight_decay = float(weight_decay)

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        theta = theta * (1.0 - self.lr * self.weight_decay)
        return super().step(theta, grad)


class Lion:
    """Lion (EvoLved Sign Momentum) optimizer."""

    def __init__(
        self,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.m: np.ndarray | None = None

    def step(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(theta)
        c = self.beta1 * self.m + (1.0 - self.beta1) * grad
        theta = theta - self.lr * np.sign(c)
        self.m = self.beta2 * self.m + (1.0 - self.beta2) * grad
        return theta


TOY_OPTIMIZERS = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "adagrad": AdaGrad,
    "rmsprop": RMSProp,
    "adam": Adam,
    "adamw": AdamW,
    "lion": Lion,
}


def build_toy_optimizer(name: str, lr: float, **kwargs):
    """Instantiate a 2D optimizer by canonical or aliased name."""
    key = name.strip().lower().replace("-", "_")
    if key == "sgd_momentum":
        key = "momentum"
    if key not in TOY_OPTIMIZERS:
        choices = ", ".join(TOY_OPTIMIZERS.keys())
        raise ValueError(f"Unknown toy optimizer {name!r}. Choices: {choices}")
    return TOY_OPTIMIZERS[key](lr, **kwargs)


def run(
    opt,
    theta0: Sequence[float] | np.ndarray,
    grad_fn: Callable[[np.ndarray], np.ndarray],
    steps: int = 50,
) -> np.ndarray:
    """Run an optimizer for a fixed number of steps on a 2D landscape.

    Returns an array of shape (N + 1, 2) recording the sequence of points.
    Stops early if coordinates become non-finite.
    """
    theta = np.asarray(theta0, dtype=float).copy()
    out = [theta.copy()]
    for _ in range(steps):
        grad = np.asarray(grad_fn(theta), dtype=float)
        theta = opt.step(theta, grad)
        out.append(theta.copy())
        if not np.all(np.isfinite(theta)):
            break
    return np.array(out)


if __name__ == "__main__":
    test_opts = [
        SGD(0.03),
        Momentum(0.006),
        Nesterov(0.008),
        AdaGrad(0.3),
        RMSProp(0.07),
        Adam(0.1),
        AdamW(0.1, weight_decay=0.02),
        Lion(0.04),
    ]
    for opt in test_opts:
        traj = run(opt, [2.8, 1.0], grad_ill_conditioned, 10)
        print(f"{type(opt).__name__:<10}: final point = {traj[-1].round(4)}")
