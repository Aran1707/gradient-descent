# Experiment Plan

## Goal

Compare optimizer behavior without accidentally changing the model, data or training budget.

## First controlled experiment

Use the toy functions first. They are more useful for teaching than raw MNIST accuracy because students can see the path.

- Function 1: `x^2+y^2`
- Function 2: `x^2+25y^2`
- Function 3: `(x^2-1)^2+0.2x+y^2`
- Function 4: `x^2-y^2`

## CNN experiment

Use your uploaded NumPy CNN.

### Recommended initial run

```bash
python train.py --epochs 3 --batch-size 64 --lr 5e-4 --train-limit 10000 --test-limit 2000
```

But for optimizer comparison, refactor optimizer logic first. Your current `step()` methods implement Adam directly inside `Conv2D` and `Dense`.

### Changes before serious comparison

1. Add validation split from training data.
2. Save CSV metrics every N steps.
3. Use validation set during training.
4. Save final test metrics only once.
5. Add optimizer interface.
6. Run multiple seeds.
7. Consider disabling dropout for the first comparison.

### Suggested optimizers and starting LR ranges

These are only starting points; tune lightly.

| Optimizer | Starting LR candidates |
|---|---:|
| SGD | 0.01, 0.03, 0.1 |
| Momentum | 0.003, 0.01, 0.03 |
| Nesterov | 0.003, 0.01, 0.03 |
| AdaGrad | 0.01, 0.03, 0.1 |
| RMSProp | 0.0005, 0.001, 0.003 |
| Adam | 0.0003, 0.001, 0.003 |
| AdamW | 0.0003, 0.001, 0.003 |
| Lion | 0.00003, 0.0001, 0.0003, or smaller than Adam |

## Plots to produce

- Training loss vs optimizer steps.
- Validation loss vs optimizer steps.
- Validation accuracy vs optimizer steps.
- Final accuracy with error bars across seeds.
- Runtime vs validation loss.

## Interpretation rules

- Do not claim an optimizer is universally best.
- Mention that optimizer comparisons depend strongly on learning rate.
- Compare both optimization speed and generalization.
- Mention compute cost: SAM uses roughly two gradient evaluations per step.
