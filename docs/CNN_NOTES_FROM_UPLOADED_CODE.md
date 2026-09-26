# Notes on the Uploaded NumPy CNN

Files copied into this package:

- `src/user_numpy_cnn/train.py`
- `src/user_numpy_cnn/model.py`

## What the code already has

From `train.py`:

- `Conv2D` with im2col/col2im.
- `MaxPool2D` forward/backward.
- `Dense` forward/backward.
- `Dropout`.
- `Flatten`.
- `ReLU`.
- `CrossEntropyLoss` with stable softmax.
- `CNNModel` assembled for MNIST.
- MNIST download with SHA-256 check.
- Training loop and `.npz` model saving.

## Architecture

```text
Input: 1×28×28
Conv2D(1→16, 3×3, padding=1) + ReLU
MaxPool2D(2×2)
Conv2D(16→32, 3×3, padding=1) + ReLU
MaxPool2D(2×2)
Dropout(0.25)
Flatten
Dense(32×7×7 → 128) + ReLU
Dropout(0.5)
Dense(128 → 10)
```

Approximate trainable parameters: 206,922.

## Optimizer separation

The trainable layers now only store parameters and backpropagated gradients.
Update state lives in `src/optimizers/`, and `CNNModel.step(optimizer)` applies
the selected rule through the shared parameter iterator.

The command-line switch selects the implementation:

```bash
python src/user_numpy_cnn/train.py --optimizer adam
python src/user_numpy_cnn/train.py --optimizer momentum
python src/user_numpy_cnn/train.py --optimizer adamw --weight-decay 1e-4
```

Supported choices include SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam,
AdamW, Lion, and SAM aliases. The training loop now keeps a validation split
for per-epoch monitoring, evaluates the official test set once at the end, and
can write structured JSON metrics and repeated-seed summaries.

## How to use this in the seminar

Use the NumPy CNN as the final application demo. It is a strong demonstration because it shows that the mathematics is not hidden behind PyTorch. But do not make the CNN the first example. Start with 2D functions and only later show MNIST.
