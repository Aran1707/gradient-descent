"""Common optimizer protocol and trainable-parameter discovery for the NumPy CNN."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

Parameter = tuple[np.ndarray, np.ndarray, str]


def iter_trainable_params(model) -> Iterator[Parameter]:
    """Yield ``(parameter, gradient, stable_name)`` triples from a model.

    Layers keep the arrays and their backpropagated gradients. Optimizers keep
    any update state, which makes the same model compatible with multiple
    optimizer implementations.
    """

    for index, layer in enumerate(model.layers):
        for name in ("W", "b"):
            parameter = getattr(layer, name, None)
            if parameter is None:
                continue
            gradient = getattr(layer, f"d{name}", None)
            if gradient is None:
                raise RuntimeError(
                    f"Missing gradient d{name} for trainable layer {index}"
                )
            yield parameter, gradient, f"layer_{index}_{name}"


class Optimizer:
    """Base class for in-place NumPy parameter updates."""

    def __init__(self, lr: float):
        if lr <= 0:
            raise ValueError("lr must be positive")
        self.lr = float(lr)

    @staticmethod
    def _materialize(params: Iterable[Parameter]) -> tuple[Parameter, ...]:
        materialized = tuple(params)
        if not materialized:
            raise ValueError("optimizer received no trainable parameters")
        for parameter, gradient, key in materialized:
            if parameter.shape != gradient.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"parameter {parameter.shape}, gradient {gradient.shape}"
                )
        return materialized

    @staticmethod
    def _state_array(
        state: dict[str, np.ndarray],
        key: str,
        parameter: np.ndarray,
    ) -> np.ndarray:
        value = state.get(key)
        if value is None:
            value = np.zeros_like(parameter)
            state[key] = value
        elif value.shape != parameter.shape:
            raise ValueError(
                f"Optimizer state shape mismatch for {key}: "
                f"state {value.shape}, parameter {parameter.shape}"
            )
        return value

    def step(self, params: Iterable[Parameter]) -> None:
        raise NotImplementedError
