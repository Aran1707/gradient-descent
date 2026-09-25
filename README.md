# Gradient Descent Seminar Package

A reusable package for a 2-hour CS115-style seminar: **gradient, gradient descent, learning rates, optimizer variants, and a NumPy CNN on MNIST**.

## What is inside

```text
PROJECT_PLAN.md              Full plan, timeline, teaching strategy
docs/                        Math notes, slide structure, talk script, experiments
figures/png/                 Ready-to-insert PNG figures
figures/svg/                 Editable vector figures
figures/gif/                 Focused optimizer animations
scripts/                     Reproducible figure generators
src/optimizers/              Shared NumPy optimizer implementations
src/landscapes/              2D objectives and optimizer trajectory code
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

## Current code entry points

The CNN now takes its update rule from `src/optimizers/` instead of keeping
Adam state inside trainable layers:

```bash
python src/user_numpy_cnn/train.py --optimizer adam
python src/user_numpy_cnn/train.py --optimizer sgd --lr 0.03
python src/user_numpy_cnn/train.py --optimizer adamw --weight-decay 1e-4
python src/user_numpy_cnn/train.py --optimizer rmsprop --disable-dropout
```

The figure generators write to `figures/generated/` by default:

```bash
python scripts/plot_landscapes.py
python scripts/plot_optimizer_trajectories.py
python scripts/plot_cnn_architecture.py
python scripts/generate_seminar_figures.py
python scripts/generate_optimizer_gifs.py --all
python scripts/generate_cnn_propagation_gif.py
```

Deterministic 2D trajectory generation:

```bash
python scripts/export_trajectories.py
```

Running unit tests and mathematical correctness checks:

```bash
python -m unittest tests/test_optim.py
```

The first three commands write focused outputs to `figures/generated/`. The
fourth regenerates all ten prepared PNG/SVG assets in `figures/png/` and
`figures/svg/`. The fifth command writes one focused GIF per optimizer to
`figures/gif/`. Use `--output-dir` on the focused commands when you want to
keep generated outputs outside the repository's prepared figure assets.
