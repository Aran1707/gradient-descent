"""Named two-dimensional objective landscapes used by the seminar."""

from __future__ import annotations

import numpy as np


def sphere(theta):
    x, y = theta
    return x * x + y * y


def grad_sphere(theta):
    x, y = theta
    return np.array([2 * x, 2 * y], dtype=float)


def ill_conditioned(theta):
    x, y = theta
    return x * x + 25 * y * y


def grad_ill_conditioned(theta):
    x, y = theta
    return np.array([2 * x, 50 * y], dtype=float)


def double_well(theta):
    x, y = theta
    return (x * x - 1) ** 2 + 0.2 * x + y * y


def grad_double_well(theta):
    x, y = theta
    return np.array([4 * x * (x * x - 1) + 0.2, 2 * y], dtype=float)


def saddle(theta):
    x, y = theta
    return x * x - y * y


def grad_saddle(theta):
    x, y = theta
    return np.array([2 * x, -2 * y], dtype=float)


OBJECTIVES = {
    "sphere": (sphere, grad_sphere),
    "ill-conditioned": (ill_conditioned, grad_ill_conditioned),
    "double-well": (double_well, grad_double_well),
    "saddle": (saddle, grad_saddle),
}
