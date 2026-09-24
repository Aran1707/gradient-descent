"""Named optimizer construction for command-line and experiment configuration."""

from __future__ import annotations

from .adaptive import AdaGrad, Adam, AdamW, Lion, RMSProp
from .first_order import Momentum, Nesterov, SGD


_ALIASES = {
    "sgd": "sgd",
    "momentum": "momentum",
    "sgd+momentum": "momentum",
    "nesterov": "nesterov",
    "adagrad": "adagrad",
    "rmsprop": "rmsprop",
    "adam": "adam",
    "adamw": "adamw",
    "lion": "lion",
}


def optimizer_names() -> tuple[str, ...]:
    """Return canonical names accepted by the CNN training CLI."""

    return tuple(_ALIASES)


def build_optimizer(name: str, lr: float, weight_decay: float = 0.0):
    """Build an optimizer from a human-readable configuration name."""

    normalized = name.strip().lower().replace("_", "-")
    canonical = _ALIASES.get(normalized)
    if canonical is None:
        choices = ", ".join(optimizer_names())
        raise ValueError(f"unknown optimizer {name!r}; choose one of: {choices}")

    if canonical == "sgd":
        return SGD(lr)
    if canonical == "momentum":
        return Momentum(lr)
    if canonical == "nesterov":
        return Nesterov(lr)
    if canonical == "adagrad":
        return AdaGrad(lr)
    if canonical == "rmsprop":
        return RMSProp(lr)
    if canonical == "adam":
        return Adam(lr)
    if canonical == "adamw":
        return AdamW(lr, weight_decay=weight_decay)
    if canonical == "lion":
        return Lion(lr)
    raise AssertionError(f"unhandled optimizer {canonical}")
