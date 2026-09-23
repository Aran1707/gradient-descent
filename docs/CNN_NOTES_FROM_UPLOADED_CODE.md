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

## Important design note for optimizer seminar

The current trainable layers implement Adam inside `step()`.

For a clean optimizer comparison, separate these responsibilities:

- Layers store parameters and gradients.
- Optimizer owns update state and performs parameter updates.

See `src/optimizer_interface_sketch.py`.

## How to use this in the seminar

Use the NumPy CNN as the final application demo. It is a strong demonstration because it shows that the mathematics is not hidden behind PyTorch. But do not make the CNN the first example. Start with 2D functions and only later show MNIST.
