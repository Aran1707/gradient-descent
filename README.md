# Gradient Descent Seminar Package

A reusable package for a 2-hour CS115-style seminar: **gradient, gradient descent, learning rates, optimizer variants, and a NumPy CNN on MNIST**.

## What is inside

```text
PROJECT_PLAN.md              Full plan, timeline, teaching strategy
docs/                        Math notes, slide structure, talk script, experiments
figures/png/                 Ready-to-insert PNG figures
figures/svg/                 Editable vector figures
src/toy/                     Toy optimizer code for 2D demos
src/user_numpy_cnn/          Your uploaded NumPy CNN files copied here
experiments/                 Suggested experiment configs
README.md                    This file
```

## Recommended seminar story

1. Teach what a gradient means geometrically.
2. Derive gradient descent from directional derivatives and Taylor approximation.
3. Use `x^2 + y^2` for intuition.
4. Use `x^2 + 25y^2` for learning-rate and curvature problems.
5. Use a separate nonconvex function for local minima.
6. Explain optimizer variants as responses to specific failure modes.
7. End with a NumPy CNN/MNIST experiment as the real model demonstration.

## Important correction

`f(x,y)=x^2+25y^2` is **not** a local-minimum trap example. It is a convex, ill-conditioned quadratic with one global minimum at `(0,0)`. Use it for oscillation, divergence, zigzagging and slow convergence. Use the double-well example in the docs for local vs global minima.
