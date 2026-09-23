# PROJECT_PLAN.md — Gradient Descent and Optimizer Variants Seminar

> This document is the handoff plan for the entire seminar package. It explains
> the teaching story, the mathematical scope, the repository structure, the
> experiment protocol, the current implementation state, and the order in
> which the remaining work should be done.

## 1. Project identity

### Course and audience

- **Course:** UIT CS115 / Mathematics for Computer Science.
- **Format:** one approximately 120-minute seminar.
- **Audience:** students who have passed calculus, probability, and linear
  algebra, but who may be new to machine learning and optimization.
- **Teaching level:** mathematically serious, but introductory in machine
  learning vocabulary.
- **Primary objective:** students should be able to explain what a gradient is,
  derive the basic gradient-descent update, and reason about why optimizer
  variants behave differently.

This is a teaching package, not only a collection of plots or a benchmark.
Every implementation and visualization should answer a classroom question.
The equations should explain the pictures, and the pictures should make the
equations easier to remember.

### Central teaching question

> Given a function, a starting point, and only local information, how can we
> move toward a lower value reliably?

The seminar begins with this question in two dimensions, where the class can
see the entire landscape. It then transfers the same ideas to a neural network
with many parameters.

### One-sentence thesis

> Gradient descent moves opposite the direction of steepest local increase;
> optimizer variants change how gradients are estimated, remembered, scaled,
> regularized, or evaluated.

## 2. Decisions carried forward from the shared seminar discussion

These decisions define the scope and prevent the seminar from becoming a list
of unrelated optimizer formulas.

1. **Teach the gradient before teaching optimizer names.** The first 44 minutes
   should establish derivatives, partial derivatives, the gradient vector,
   directional derivatives, and the basic update rule.
2. **Use the same function in several representations.** Students should see
   an equation, a gradient field, a 3D surface, a contour plot, and an
   optimization trajectory.
3. **Keep failure modes mathematically distinct.**
   - `x^2 + 25y^2` is convex and has one global minimum. It demonstrates
     anisotropic curvature, zigzagging, slow progress, oscillation, and
     divergence.
   - A separate nonconvex double-well function demonstrates local versus
     global minima.
   - `x^2 - y^2` demonstrates that a zero gradient does not always identify a
     minimum.
4. **Introduce variants as responses to problems.** For example, momentum
   addresses persistent directions and oscillation; RMSProp addresses
   coordinate-wise scale differences; Adam combines momentum-like averaging
   with adaptive scaling.
5. **Use the NumPy CNN as the application, not as the opening example.** The
   CNN demonstrates that backpropagation computes gradients while the
   optimizer decides how to use them.
6. **Keep advanced optimizers in proportion to the lecture.** Nesterov belongs
   in the main story because it is a natural momentum extension. Lion, SAM,
   L-BFGS, and Muon are appendix or bonus material unless the instructor
   explicitly requests more modern optimizer coverage.
7. **Do not claim that a toy trajectory proves universal optimizer superiority.**
   Toy surfaces teach mechanisms; controlled MNIST experiments show how those
   mechanisms behave in one concrete model and data setting.

## 3. Learning objectives

By the end of the seminar, a student should be able to:

1. Distinguish a derivative, a partial derivative, and a gradient.
2. Compute `∇f(x, y)` for a simple function by hand.
3. Explain why the gradient points in the direction of steepest local increase.
4. Derive and apply
   `theta_(t+1) = theta_t - eta * grad f(theta_t)`.
5. Predict how a learning rate that is too small, appropriate, or too large
   changes an optimization trajectory.
6. Explain why `x^2 + 25y^2` is ill-conditioned rather than a local-minimum
   trap.
7. Distinguish a local minimum, a global minimum, and a saddle point.
8. Explain the conceptual difference between full-batch, stochastic, and
   mini-batch gradients.
9. Describe what state is maintained by momentum, AdaGrad, RMSProp, Adam,
   and AdamW.
10. Explain the separation between backpropagation and parameter updates in the
    NumPy CNN.
11. Interpret training loss, validation loss, accuracy, runtime, and
    variability across random seeds without declaring a universal winner.

## 4. Package structure

The package is intentionally split into directions, reusable source code,
rendered teaching assets, and generated experiment outputs.

```text
gradient-descent/
├── PROJECT_PLAN.md
├── README.md
├── PROJECT_TREE.txt
├── docs/
│   ├── MATH_NOTES.md
│   ├── OPTIMIZER_CHEATSHEET.md
│   ├── SLIDE_STRUCTURE.md
│   ├── TALK_SCRIPT.md
│   ├── QUESTIONS_FOR_CLASS.md
│   ├── EXPERIMENT_PLAN.md
│   ├── FIGURE_INDEX.md
│   ├── CNN_NOTES_FROM_UPLOADED_CODE.md
│   └── SOURCES.md
├── figures/
│   ├── png/
│   └── svg/
├── experiments/
│   └── mnist_experiment_plan.json
├── data/
│   └── generated trajectories and metrics; intentionally empty after cleanup
├── assets/
│   └── reserved for slide/deck assets; currently empty
└── src/
    ├── optimizer_interface_sketch.py
    ├── toy/
    │   └── toy_optimizers.py
    └── user_numpy_cnn/
        ├── model.py
        └── train.py
```

### File and directory responsibilities

| Path | Responsibility | Current status |
|---|---|---|
| `PROJECT_PLAN.md` | Full narrative, scope, workflow, and acceptance criteria | This handoff document |
| `README.md` | Short orientation for someone opening the package | Present |
| `PROJECT_TREE.txt` | Human-readable inventory of the package | Updated to match cleanup |
| `docs/MATH_NOTES.md` | Hand derivations and mathematical correctness notes | Present |
| `docs/OPTIMIZER_CHEATSHEET.md` | Compact update rules and optimizer intuition | Present |
| `docs/SLIDE_STRUCTURE.md` | Proposed main-deck and appendix order | Present |
| `docs/TALK_SCRIPT.md` | Speaker wording and emphasis | Present |
| `docs/QUESTIONS_FOR_CLASS.md` | Prediction exercises and discussion prompts | Present |
| `docs/EXPERIMENT_PLAN.md` | Short experiment protocol and learning-rate starting points | Present |
| `docs/FIGURE_INDEX.md` | Mapping from figure names to teaching purposes | Present |
| `docs/CNN_NOTES_FROM_UPLOADED_CODE.md` | Notes about the uploaded NumPy CNN | Present |
| `docs/SOURCES.md` | Further reading and external references | Present |
| `figures/png/` | Slide-ready raster figures | Preserved reusable assets |
| `figures/svg/` | Editable vector versions of the figures | Preserved reusable assets |
| `experiments/mnist_experiment_plan.json` | Machine-readable experiment policy | Present |
| `data/` | Regenerated trajectories, metrics, and run outputs | Empty by design after cleanup |
| `src/toy/toy_optimizers.py` | Small, visible 2D optimizer simulator | Partial prototype |
| `src/optimizer_interface_sketch.py` | Proposed common parameter/gradient optimizer interface | Educational sketch |
| `src/user_numpy_cnn/train.py` | Original NumPy CNN, data loading, training, and inference model saving | Preserved original implementation |
| `src/user_numpy_cnn/model.py` | Inference wrapper for saved CNN models and handwritten images | Preserved original implementation |

The empty `data/` directory is not a missing source file. It is the output
location for future runs. The instructions and source needed to regenerate the
outputs remain in the package.

## 5. Cleanup and data policy

### What was removed

The following generated toy trajectories were deleted because they can be
regenerated and should not be mistaken for final experimental evidence:

```text
data/traj_adagrad.csv
data/traj_adam.csv
data/traj_adamw.csv
data/traj_gd.csv
data/traj_lion.csv
data/traj_momentum.csv
data/traj_nesterov.csv
data/traj_rmsprop.csv
```

Each file contained only an `x,y` trajectory from a toy optimizer run. No
teaching directions, source code, mathematical notes, experiment plans, or
figures were removed.

### What was deliberately preserved

- All markdown directions and seminar notes.
- The generated PNG and SVG figures, including the architecture diagram and
  the training-loss template.
- The original NumPy CNN source.
- The toy optimizer source and optimizer-interface sketch.
- The machine-readable MNIST experiment plan.
- The source and checksum information used to download MNIST.

The preserved figures are reusable teaching assets, but any figure derived from
the deleted trajectories should be treated as preliminary until the toy
experiments are regenerated and checked against the current implementation.
The `10_training_loss_template` figure is a template, not a result.

### Generated-output rule

Future runs should keep a clear distinction:

- **Source/directions:** code, markdown, JSON configuration, and editable SVG
  assets.
- **Generated outputs:** CSV/NPZ metrics, toy trajectories, rendered result
  plots, downloaded datasets, model checkpoints, and run logs.
- **Final presentation assets:** figures or tables selected only after the
  corresponding run has passed the reproducibility and evaluation checks.

Every generated run should record, at minimum:

- optimizer name and hyperparameters;
- function or model name;
- initial point or initialization seed;
- data split and batch size;
- number of optimizer steps or epochs;
- software/runtime information;
- output filenames;
- whether the result is illustrative, a smoke test, a screening run, or a
  final comparison.

## 6. The 120-minute seminar plan

The main deck should contain approximately 28–35 slides. Derivations,
alternative optimizer formulas, and implementation details can go in an
appendix so the live lecture remains teachable.

| Time | Section | Classroom purpose | Required output |
|---:|---|---|---|
| 0–8 min | Why optimization? | Introduce losses, parameters, and the search problem | One motivating example |
| 8–28 min | What is a gradient? | Connect derivative → partial derivatives → vector field | Hand calculation and vector-field figure |
| 28–44 min | Derive gradient descent | Use directional derivatives, Cauchy–Schwarz, and Taylor approximation | Update rule plus one hand-computed step |
| 44–59 min | Curvature and learning rate | Explain slow progress, oscillation, and divergence | Ill-conditioned quadratic and learning-rate comparison |
| 59–64 min | Prediction exercise | Make students compute or predict the next point | Reveal the answer after discussion |
| 64–79 min | SGD and momentum | Separate gradient estimation from gradient transformation | Mini-batch formula and momentum trajectory |
| 79–96 min | Adaptive optimizers | Explain coordinate-wise scaling and bias correction | AdaGrad, RMSProp, Adam, and AdamW |
| 96–108 min | Toy optimizer comparison | Compare identical starts, surfaces, axes, and budgets | Synchronized contour plots |
| 108–114 min | NumPy CNN application | Connect backpropagation to optimizer updates | Architecture and code-level separation |
| 114–120 min | Takeaways and Q&A | Check understanding and discuss limitations | Summary slide and questions |

Advanced topics such as Lion, SAM, L-BFGS, Muon, full convergence proofs,
and detailed CNN convolution backpropagation should be prepared as appendix
material. They can be shown when questions arise without displacing the
gradient-first explanation.

## 7. Mathematical teaching sequence

### 7.1 Start with a one-dimensional reminder

Use `f(x) = x^2` to establish that the derivative describes local change.
At `x > 0`, the function increases as `x` increases; at `x < 0`, it decreases
as `x` increases. The class already knows this calculus, so the goal is to
activate the idea rather than reteach one-variable differentiation.

### 7.2 Move to the gradient

For

```text
f(x, y) = x^2 + y^2
```

the partial derivatives are

```text
∂f/∂x = 2x
∂f/∂y = 2y
∇f(x, y) = (2x, 2y)
```

Emphasize that the gradient is a vector in the input/parameter space. It is
not another scalar derivative and it is not automatically the same object as a
normal vector to a plotted 3D surface, even though the two ideas are related
through the graph of the function.

At `(1, 2)`,

```text
∇f(1, 2) = (2, 4).
```

With `eta = 0.1`,

```text
theta_1 = (1, 2) - 0.1(2, 4) = (0.8, 1.6).
```

The function value drops from `5` to `0.8^2 + 1.6^2 = 3.2`. This is the first
place where students should see the equation, the vector, and the visual
trajectory agree.

### 7.3 Explain why the negative gradient is downhill

For a unit direction `u`, the directional derivative is

```text
D_u f = ∇f · u.
```

By Cauchy–Schwarz,

```text
|∇f · u| ≤ ||∇f|| ||u|| = ||∇f||.
```

The largest directional derivative occurs when `u` points with the gradient;
the most negative one occurs when `u` points against it. This is the
mathematical reason for the negative-gradient direction.

Then use the first-order Taylor approximation:

```text
f(theta + delta) ≈ f(theta) + ∇f(theta)^T delta.
```

Setting `delta = -eta ∇f(theta)` gives

```text
f(theta - eta ∇f(theta))
  ≈ f(theta) - eta ||∇f(theta)||^2.
```

This predicts local decrease for a sufficiently small positive learning rate.
It does not guarantee that every arbitrarily large step will decrease the
function.

### 7.4 Use the ill-conditioned quadratic correctly

Use

```text
f(x, y) = x^2 + 25y^2.
```

Its gradient and Hessian are

```text
∇f(x, y) = (2x, 50y)
H = [[2, 0],
     [0, 50]].
```

The eigenvalues are `2` and `50`, so the Hessian condition number is `25`.
The contours are narrow ellipses: curvature is much stronger in the `y`
direction than in the `x` direction.

The coordinate update is

```text
x_(t+1) = (1 - 2eta)x_t
y_(t+1) = (1 - 50eta)y_t.
```

For both coordinates to converge, both multiplier magnitudes must be below
one. The stricter condition is

```text
0 < eta < 0.04.
```

Use this example to show:

- a very small step: stable but slow in the shallow direction;
- a moderate step: faster progress;
- a larger step: alternating signs and oscillation in the steep direction;
- a still larger step: divergence.

State explicitly:

> This function is convex and has one global minimum at `(0, 0)`. It is not a
> local-minimum-trapping example.

### 7.5 Add separate nonconvex and saddle examples

For local/global minima, use a nonconvex function such as

```text
f(x, y) = (x^2 - 1)^2 + 0.2x + y^2.
```

Different initial positions can fall into different basins. The lower well is
on the negative-`x` side, while the positive side can contain a higher local
minimum.

For a saddle point, use

```text
f(x, y) = x^2 - y^2,
∇f(x, y) = (2x, -2y).
```

At the origin the gradient is zero, but the point is neither a local minimum
nor a local maximum. This prevents the common misconception that “zero
gradient” automatically means “finished at a minimum.”

## 8. Optimizer scope and teaching roles

The seminar should introduce the problem before the formula.

| Optimizer/family | Main mathematical idea | Problem it addresses | Lecture role |
|---|---|---|---|
| Batch GD | Use the full gradient | Baseline deterministic update | Derive carefully |
| Mini-batch SGD | Estimate the gradient from a batch | Full gradients can be expensive | Explain carefully |
| Momentum | Accumulate a velocity-like state | Noise and zigzagging | Derive and visualize |
| Nesterov | Evaluate at a look-ahead position | Improve momentum correction | Main-story extension |
| AdaGrad | Accumulate squared gradients per coordinate | Different coordinates have different scales | Explain |
| RMSProp | Exponentially average squared gradients | AdaGrad's accumulator can shrink steps forever | Derive compactly |
| Adam | First moment + second moment + bias correction | Combine momentum and adaptive scaling | Give the most detail |
| AdamW | Decouple weight decay | Adaptive scaling changes the effect of an L2 penalty | Explain precisely |
| Lion | Sign-based momentum update | Explore lower-state, magnitude-independent updates | Bonus/advanced |
| SAM | Optimize a neighborhood around the point | Encourage flatter local regions | Bonus; costly |
| L-BFGS | Approximate curvature without a full Hessian | Use second-order information more cheaply | Appendix |
| Muon | Matrix-parameter update geometry | Specialized modern optimizer idea | Appendix only |

Important distinctions:

- Batch size describes how the gradient is estimated.
- Momentum, adaptive methods, and weight decay describe how an estimate is
  transformed or combined with optimizer state.
- A common nominal learning rate is useful for demonstrating update mechanics,
  but it is not automatically a fair performance comparison.
- Tuned comparisons must state the tuning budget and search procedure.

## 9. Visualization plan

Each figure should have one teaching job.

| Asset | Teaching job | Intended use |
|---|---|---|
| `01_gradient_field_sphere` | Arrows show the gradient at each point | Gradient intuition |
| `02_surface_sphere_3d` | Show the bowl and its minimum | Connect equation to surface |
| `03_gd_steps_sphere` | Show successive negative-gradient updates | First trajectory |
| `04_learning_rate_ill_conditioned` | Compare learning-rate regimes | Curvature and stability |
| `05_optimizer_trajectories_ill_conditioned` | Compare paths on identical axes | Optimizer behavior |
| `06_nonconvex_local_minima` | Show distinct basins and wells | Local versus global minima |
| `07_saddle_point` | Show a stationary point that is not a minimum | Failure of the zero-gradient shortcut |
| `08_optimizer_family_map` | Organize variants by the problem they address | Transition slide |
| `09_numpy_cnn_architecture` | Explain the final model at a high level | Application |
| `10_training_loss_template` | Provide a visual style for future results | Placeholder only |

For optimizer comparisons, use the same initial point, objective, axes,
parameter budget, and stopping rule. Small multiples are often clearer than
putting too many overlapping paths on one contour plot.

Animations should be exported before the seminar. The live lecture should not
depend on an internet connection, a remote notebook, or a browser demo.

## 10. NumPy CNN application

### Current model

The preserved model in `src/user_numpy_cnn/train.py` is:

```text
Input              1 × 28 × 28
Conv2D + ReLU      1 → 16 channels, 3 × 3, padding 1
MaxPool2D          2 × 2
Conv2D + ReLU      16 → 32 channels, 3 × 3, padding 1
MaxPool2D          2 × 2
Dropout            rate 0.25
Flatten
Dense + ReLU       32 × 7 × 7 → 128
Dropout            rate 0.50
Dense              128 → 10
```

The model has approximately 206,922 trainable parameters and implements
convolution with `im2col`/`col2im`, pooling, dense layers, dropout, ReLU,
cross-entropy, MNIST loading, and `.npz` model saving.

### Current implementation limitations

The original implementation is intentionally preserved, but it is not yet a
clean multi-optimizer experiment engine:

- `Conv2D` and `Dense` own Adam-like state and update themselves inside
  `step()`.
- The model evaluates on the official test data after every epoch.
- There is no train/validation split in the current training loop.
- The current loop prints selected batch losses and epoch test accuracy but
  does not write structured per-step CSV/NPZ metrics.
- Dropout is enabled during normal training, which adds stochastic variation
  to optimizer comparisons.
- The current code decays the learning rate by multiplying it by `0.95` after
  each epoch.
- The default MNIST cache is `src/user_numpy_cnn/data/mnist.npz`, not the
  seminar-level `data/` directory.
- NumPy operations do not automatically use the machine's NVIDIA GPU.

The `src/optimizer_interface_sketch.py` file shows the intended separation:
layers expose parameters and gradients, while the optimizer owns state and
updates parameters. It currently sketches SGD, Momentum, and Adam; it is not
yet the completed training backend.

### Educational message

The CNN section should show this separation:

```text
forward pass      → compute predictions and loss
backpropagation   → compute parameter gradients
optimizer.step()  → transform gradients and update parameters
```

Do not spend the main lecture explaining every convolution indexing detail.
Show the architecture, one update path, and the optimizer interface. Keep
full convolution derivations and gradient checks in the appendix or repository.

## 11. Controlled experiment protocol

### Stage 0 — correctness and smoke checks

Before comparing optimizers:

1. Check the analytical gradient of the toy functions against finite
   differences where appropriate.
2. Verify that the simple bowl decreases for a small stable learning rate.
3. Verify that the ill-conditioned example exhibits the predicted oscillation
   and divergence boundaries.
4. Verify that each optimizer preserves parameter shapes and does not mutate
   unrelated state.
5. Run a tiny CNN forward/backward smoke test and check that loss is finite.

### Stage 1 — toy surfaces

Use:

```text
f1(x, y) = x^2 + y^2
f2(x, y) = x^2 + 25y^2
f3(x, y) = (x^2 - 1)^2 + 0.2x + y^2
f4(x, y) = x^2 - y^2
```

For each comparison, record:

- function name;
- initial point;
- optimizer and hyperparameters;
- number of steps;
- coordinates at every step;
- objective value at every step;
- termination or divergence status.

These outputs belong in `data/` and should be generated by source code rather
than manually edited.

### Stage 2 — small CNN screening

Use the NumPy CNN on a stratified or documented subset of MNIST. The existing
experiment configuration proposes:

```text
training samples: 10,000
validation samples: 2,000
batch size:       64
epochs:           3
seeds:            0, 1, 2
```

For the first controlled comparison, disable dropout so the comparison focuses
on optimizer behavior. Later, repeat with the original dropout configuration
for a more realistic demonstration.

### Stage 3 — focused comparison

After screening, select a smaller set of optimizers for longer runs. A sensible
main comparison is SGD, Momentum, RMSProp, Adam, and AdamW, with Nesterov as
an optional extension. Reserve Lion and SAM for bonus runs because their
hyperparameters and compute costs need separate explanation.

### Comparison controls

For corresponding runs:

- use the same model architecture;
- use the same train/validation split;
- use the same initialization for each seed;
- use the same batch ordering for each seed;
- use the same number of optimizer steps;
- standardize inputs using training-split statistics if standardization is
  introduced;
- keep the test set untouched until the final evaluation;
- report the actual learning-rate choices;
- report multiple seeds rather than the most attractive single run.

Run two kinds of comparison:

1. **Common nominal learning rate:** useful for showing that optimizers
   transform the same gradient differently.
2. **Lightly tuned learning rates:** more informative for practical behavior,
   but only if the tuning budget is stated.

### Metrics

Save and plot:

- training loss versus optimizer step;
- validation loss versus optimizer step;
- validation accuracy versus optimizer step;
- final test accuracy, evaluated once after model selection;
- runtime versus validation loss or accuracy;
- mean and variation across seeds.

Do not treat lower training loss as automatic evidence of better
generalization. Do not treat the fastest wall-clock run as the best optimizer
without also reporting the quality reached.

## 12. Implementation roadmap

The order below keeps the mathematics and experiments aligned.

### Phase A — Rebuild the toy pipeline

1. Keep `src/toy/toy_optimizers.py` as the small, readable teaching module.
2. Add the missing toy optimizer implementations only when their classroom
   role is defined.
3. Add the nonconvex and saddle objectives.
4. Add deterministic trajectory export with explicit configuration.
5. Regenerate CSV trajectories into `data/`.
6. Regenerate the affected plots and check each against the equations.

### Phase B — Separate CNN parameters from optimization

1. Define a common parameter/gradient iterator.
2. Move optimizer state out of `Conv2D` and `Dense`.
3. Implement optimizer objects with explicit state keyed to parameters.
4. Preserve model forward and backward behavior.
5. Add focused finite-difference and shape checks.
6. Confirm that the Adam refactor matches the original behavior within an
   explicitly documented tolerance.

### Phase C — Make evaluation scientifically usable

1. Add a train/validation split.
2. Keep official test data for final evaluation.
3. Add structured metrics output.
4. Add a run configuration/manifest.
5. Add repeated-seed execution.
6. Support dropout-disabled controlled runs.
7. Keep model checkpoints and MNIST cache outside the seminar toy-output
   directory unless a run explicitly needs to package them.

### Phase D — Produce the seminar package

1. Select final toy trajectories after checks pass.
2. Select a small, defensible CNN comparison.
3. Render training and validation plots.
4. Write the slide deck and speaker notes around the verified results.
5. Add appendix derivations and implementation details.
6. Export a PDF and offline copies of animations.
7. Rehearse the prediction exercises and timing.

## 13. Reproducibility and handoff rules

A new contributor should be able to answer these questions before running
anything:

1. What is the mathematical example?
2. What is the initial point?
3. Which optimizer and hyperparameters are being used?
4. How many steps or epochs are allowed?
5. Which split is training, validation, and test?
6. Which outputs will be created?
7. Which result is illustrative, and which result is final evidence?

Recommended run metadata:

```text
run_id
timestamp
source revision or file hashes
Python and NumPy versions
objective/model name
optimizer and hyperparameters
seed
data split
batch size
step/epoch budget
output paths
status and notes
```

The MNIST loader already verifies the downloaded archive using the SHA-256
constant in `src/user_numpy_cnn/train.py`. Preserve that check if the loader
is changed.

## 14. Deliverables

### Main teaching deliverables

- A 28–35-slide main deck.
- Speaker notes or a talk script.
- A clear derivation of the gradient and gradient-descent rule.
- The four core landscape examples.
- Synchronized optimizer visualizations.
- A short NumPy CNN/MNIST application.
- Prediction questions and discussion prompts.

### Technical deliverables

- Reproducible toy trajectory generator.
- Common optimizer interface for the CNN.
- Controlled MNIST experiment configuration.
- Metrics files with run metadata.
- Training/validation/test plots.
- Final result summary with seed variation and runtime.
- Offline backup of the deck and animations.

### Appendix deliverables

- General quadratic convergence:
  `theta_(t+1) = (I - eta H)theta_t`.
- Linear-regression gradient derivation.
- Adam bias-correction derivation.
- Nesterov variants.
- SAM pseudocode and compute cost.
- CNN backpropagation overview.
- Finite-difference gradient checking.

## 15. Acceptance criteria

The package is ready for presentation when:

- the audience can follow the gradient before seeing Adam;
- the ill-conditioned example is not described as local-minimum trapping;
- the nonconvex and saddle examples are visibly separate;
- each main optimizer is introduced as a response to a concrete problem;
- toy plots use verified, reproducible trajectories;
- the CNN comparison separates backpropagation from optimizer updates;
- test data is not used as an every-epoch tuning signal;
- metrics include validation behavior and multiple seeds;
- the presentation states that optimizer rankings depend on hyperparameters,
  model, data, and compute budget;
- the deck can run offline from prepared assets;
- a new contributor can regenerate outputs from the documented source and
  configuration.

## 16. Risks and explicit non-goals

### Risks

- Too many optimizer formulas can overwhelm beginners.
- A single seed can make noise look like a real optimizer difference.
- A common learning rate can unfairly favor one optimizer.
- A 2D trajectory can be overinterpreted as evidence about a
  high-dimensional neural-network loss.
- NumPy CNN training can be slow, especially with SAM or many seeds.
- Test-set evaluation during development can leak selection information.
- Live internet demos can fail during the seminar.

### Non-goals for the main lecture

- Proving every stochastic-optimization convergence theorem.
- Explaining every convolution implementation line.
- Declaring one optimizer universally best.
- Implementing every modern optimizer in the first pass.
- Turning the seminar into a general neural-network architecture lecture.

The quality bar is a coherent, defensible teaching story with reproducible
evidence—not the largest possible optimizer catalog.
