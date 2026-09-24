"""Small optimizer trajectory runner for two-dimensional landscapes."""

import numpy as np

try:
    from .objectives import (
        grad_ill_conditioned,
        grad_sphere,
        ill_conditioned,
        sphere,
    )
except ImportError:  # Supports running this file directly as a teaching script.
    from objectives import grad_ill_conditioned, grad_sphere, ill_conditioned, sphere


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, theta, grad):
        return theta - self.lr * grad


class Momentum:
    def __init__(self, lr, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, theta, grad):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.beta * self.v + grad
        return theta - self.lr * self.v


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad * grad)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def run(opt, theta0, grad_fn, steps=50):
    theta = np.asarray(theta0, dtype=float)
    out = [theta.copy()]
    for _ in range(steps):
        theta = opt.step(theta, grad_fn(theta))
        out.append(theta.copy())
    return np.array(out)


if __name__ == "__main__":
    for opt in [SGD(0.03), Momentum(0.006), Adam(0.1)]:
        print(type(opt).__name__, run(opt, [2.8, 1.0], grad_ill_conditioned, 10)[-1])
