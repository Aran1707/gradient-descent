"""Named optimizer construction for command-line and experiment configuration."""

from __future__ import annotations

from .adaptive import AdaGrad, Adam, AdamW, Lion, RMSProp
from .base import Optimizer
from .first_order import Momentum, Nesterov, SGD
from .sam import SAM


_ALIASES = {
    "sgd": "sgd",
    "momentum": "momentum",
    "sgd+momentum": "momentum",
    "sgd-momentum": "momentum",
    "nesterov": "nesterov",
    "adagrad": "adagrad",
    "rmsprop": "rmsprop",
    "adam": "adam",
    "adamw": "adamw",
    "lion": "lion",
    "sam": "sam-sgd",
    "sam-sgd": "sam-sgd",
    "sam-momentum": "sam-momentum",
    "sam-adam": "sam-adam",
}


def optimizer_names() -> tuple[str, ...]:
    """Return canonical names accepted by the CNN training CLI."""
    return tuple(_ALIASES)


def build_sam_optimizer(
    base: str | Optimizer,
    lr: float = 1e-3,
    rho: float = 0.05,
    **kwargs,
) -> SAM:
    """Build a SAM wrapper around a named or existing optimizer."""
    if isinstance(base, str):
        base_opt = build_optimizer(base, lr=lr, **kwargs)
    elif isinstance(base, Optimizer):
        base_opt = base
    else:
        raise TypeError("base must be a string optimizer name or an Optimizer instance")
    return SAM(base_opt, rho=rho)


def build_optimizer(
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    rho: float = 0.05,
):
    """Build an optimizer from a human-readable configuration name."""
    normalized = name.strip().lower().replace("_", "-").replace("+", "-")
    canonical = _ALIASES.get(normalized)
    if canonical is None:
        choices = ", ".join(optimizer_names())
        raise ValueError(f"unknown optimizer {name!r}; choose one of: {choices}")

    if canonical.startswith("sam-"):
        base_name = canonical[4:]
        base_opt = build_optimizer(base_name, lr, weight_decay=weight_decay)
        return SAM(base_opt, rho=rho)

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
