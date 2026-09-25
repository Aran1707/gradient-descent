"""Sharpness-Aware Minimization (SAM) optimizer wrapper."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .base import Optimizer, Parameter


class SAM:
    """Sharpness-Aware Minimization wrapper around a base Optimizer.

    SAM simultaneously minimizes loss value and loss sharpness by seeking
    parameters whose local neighborhood has uniformly low loss:
        min_w max_{||e||_2 <= rho} L(w + e)

    The standard two-pass training step proceeds as:
        1. Compute gradient g_1 = \nabla L(w).
        2. Call sam.first_step(model.parameters()) to perturb parameters:
               e = rho * g_1 / (||g_1||_2 + eps)
               w <- w + e
        3. Compute gradient g_2 = \nabla L(w + e).
        4. Call sam.second_step(model.parameters()) to restore and update:
               w <- w - e
               base_optimizer.step(...) using g_2
    """

    def __init__(self, base_optimizer: Optimizer, rho: float = 0.05, eps: float = 1e-12):
        if not isinstance(base_optimizer, Optimizer):
            raise TypeError("base_optimizer must be an instance of Optimizer")
        if rho <= 0:
            raise ValueError("rho must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self.eps = float(eps)
        self.perturbations: dict[str, np.ndarray] = {}

    @property
    def lr(self) -> float:
        return self.base_optimizer.lr

    @lr.setter
    def lr(self, value: float) -> None:
        self.base_optimizer.lr = float(value)

    def first_step(self, params: Iterable[Parameter]) -> None:
        """Perturb parameters along the normalized gradient direction."""
        materialized = self.base_optimizer._materialize(params)

        total_norm_sq = sum(
            float(np.sum(gradient * gradient))
            for _, gradient, _ in materialized
        )
        norm = np.sqrt(total_norm_sq) + self.eps
        scale = self.rho / norm

        for parameter, gradient, key in materialized:
            e = gradient * scale
            self.perturbations[key] = e
            parameter[...] += e

    def second_step(self, params: Iterable[Parameter]) -> None:
        """Restore parameters from perturbation and take base optimizer step."""
        materialized = self.base_optimizer._materialize(params)
        for parameter, _, key in materialized:
            if key not in self.perturbations:
                raise RuntimeError(f"Missing SAM perturbation for parameter {key}")
            parameter[...] -= self.perturbations[key]
        self.perturbations.clear()
        self.base_optimizer.step(materialized)

    def step(self, params: Iterable[Parameter]) -> None:
        """Standard step without perturbation (delegates to base optimizer)."""
        self.base_optimizer.step(params)
