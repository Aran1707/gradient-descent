"""Sketch for refactoring the uploaded NumPy CNN to support multiple optimizers.

This is intentionally separate from train.py so your original flex stays intact.
"""
from dataclasses import dataclass, field
import numpy as np

def iter_trainable_params(model):
    """Yield parameter arrays and gradient arrays from trainable layers."""
    for layer in model.layers:
        if hasattr(layer, "W"):
            yield layer.W, layer.dW, f"{id(layer)}.W"
            yield layer.b, layer.db, f"{id(layer)}.b"

@dataclass
class SGD:
    lr: float
    def step(self, params):
        for p, g, _ in params:
            p -= self.lr * g

@dataclass
class Momentum:
    lr: float
    beta: float = 0.9
    state: dict = field(default_factory=dict)
    def step(self, params):
        for p, g, key in params:
            v = self.state.setdefault(key, np.zeros_like(p))
            v[...] = self.beta * v + g
            p -= self.lr * v

@dataclass
class Adam:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)
    def step(self, params):
        self.t += 1
        for p, g, key in params:
            m = self.m.setdefault(key, np.zeros_like(p))
            v = self.v.setdefault(key, np.zeros_like(p))
            m[...] = self.beta1 * m + (1 - self.beta1) * g
            v[...] = self.beta2 * v + (1 - self.beta2) * (g * g)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
