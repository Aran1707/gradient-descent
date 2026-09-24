import numpy as np
import hashlib
from pathlib import Path
import sys
import time
from urllib.request import urlretrieve


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optimizers import build_optimizer, iter_trainable_params, optimizer_names


DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_MODEL_PATH = PROJECT_DIR / "cnn-model.npz"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
MODEL_FORMAT_VERSION = 1
DTYPE = np.float32


# core engine: im2col & col2im
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    if len(x_shape) != 4:
        raise ValueError("x_shape must be (N, C, H, W)")
    if field_height <= 0 or field_width <= 0:
        raise ValueError("field dimensions must be positive")
    if padding < 0 or stride <= 0:
        raise ValueError("padding cannot be negative and stride must be positive")

    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1
    if out_height <= 0 or out_width <= 0:
        raise ValueError("field dimensions do not fit the input shape")

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]


# nn layers
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        if min(in_channels, out_channels, kernel_size) <= 0:
            raise ValueError("channel counts and kernel_size must be positive")
        if stride <= 0 or padding < 0:
            raise ValueError("stride must be positive and padding cannot be negative")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # he initialization
        self.W = (
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
            * np.sqrt(2.0 / (in_channels * kernel_size**2))
        ).astype(DTYPE)
        self.b = np.zeros((out_channels, 1), dtype=DTYPE)

    def forward(self, X):
        self.X = X
        n, c, h, w = X.shape
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.X_col = im2col_indices(
            X, self.kernel_size, self.kernel_size, self.padding, self.stride
        )
        W_col = self.W.reshape(self.out_channels, -1)

        out = W_col @ self.X_col + self.b

        return out.reshape(self.out_channels, out_h, out_w, n).transpose(3, 0, 1, 2)

    def backward(self, dout):
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        self.dW = (dout_reshaped @ self.X_col.T).reshape(self.W.shape)
        self.db = np.sum(dout_reshaped, axis=1, keepdims=True)

        W_reshape = self.W.reshape(self.out_channels, -1)
        dX_col = W_reshape.T @ dout_reshaped

        return col2im_indices(
            dX_col,
            self.X.shape,
            self.kernel_size,
            self.kernel_size,
            self.padding,
            self.stride,
        )


class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        if pool_size <= 0 or stride <= 0:
            raise ValueError("pool_size and stride must be positive")
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        n, c, h, w = X.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        X_reshaped = X.reshape(n * c, 1, h, w)
        self.X_col = im2col_indices(
            X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride
        )

        self.max_indices = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_indices, np.arange(self.X_col.shape[1])]

        return out.reshape(out_h, out_w, n, c).transpose(2, 3, 0, 1)

    def backward(self, dout):
        n, c, h, w = self.X.shape
        dX_col = np.zeros_like(self.X_col)
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        dX_col[self.max_indices, np.arange(self.X_col.shape[1])] = dout_flat

        dX = col2im_indices(
            dX_col,
            (n * c, 1, h, w),
            self.pool_size,
            self.pool_size,
            padding=0,
            stride=self.stride,
        )
        return dX.reshape(self.X.shape)


class Dense:
    def __init__(self, in_features, out_features):
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature counts must be positive")
        self.W = (
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        ).astype(DTYPE)
        self.b = np.zeros(out_features, dtype=DTYPE)

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout):
        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T


class Dropout:
    def __init__(self, rate=0.5):
        """
        rate: The probability of dropping a neuron (e.g., 0.5 means drop 50%).
        """
        if not 0.0 <= rate < 1.0:
            raise ValueError("dropout rate must be in the range [0, 1)")
        self.rate = rate
        self.mask = None
        self.training = True  # flag to toggle between train/test modes

    def forward(self, X):
        if self.training:
            # create a binary mask using a binomial distribution
            # scale by 1 / (1 - rate) to keep the expected value of the activations consistent
            self.mask = np.random.binomial(1, 1 - self.rate, size=X.shape).astype(
                DTYPE
            ) / (1.0 - self.rate)

            return X * self.mask
        else:
            # during inference -> sleep
            self.mask = None
            return X

    def backward(self, dout):
        # gradients only flow through the neurons that were kept active
        return dout if self.mask is None else dout * self.mask


class Flatten:
    def forward(self, X):
        self.X_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.X_shape)


class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dout):
        return dout * (self.X > 0)


class CrossEntropyLoss:
    def forward(self, logits, y):
        if logits.ndim != 2 or y.ndim != 1 or logits.shape[0] != y.shape[0]:
            raise ValueError("logits must be 2-D and y must match its batch dimension")
        m = y.shape[0]
        if m == 0 or np.any(y < 0) or np.any(y >= logits.shape[1]):
            raise ValueError("labels must be valid class indices for a non-empty batch")
        # shift logits for numerical stability
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        log_probs = -np.log(self.probs[np.arange(m), y] + 1e-15)

        return np.sum(log_probs) / m

    def backward(self, y):
        m = y.shape[0]
        dout = self.probs.copy()
        dout[np.arange(m), y] -= 1

        return dout / m


# network assembly n trainning
class CNNModel:
    def __init__(self, use_dropout=True):
        self.training = True
        self.use_dropout = bool(use_dropout)
        spatial_dropout = 0.25 if self.use_dropout else 0.0
        dense_dropout = 0.5 if self.use_dropout else 0.0
        self.layers = [
            Conv2D(1, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Conv2D(16, 32, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Dropout(rate=spatial_dropout),  # drop 25% of spatial features
            Flatten(),
            Dense(32 * 7 * 7, 128),
            ReLU(),
            Dropout(
                rate=dense_dropout
            ),  # drop 50% of dense features to prevent memorization
            Dense(128, 10),  # 10 digits 0 -> 9
        ]
        self.loss_fn = CrossEntropyLoss()

    def train(self):
        """Sets the model to training mode (enables Dropout)."""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self):
        """Sets the model to evaluation mode (disables Dropout)."""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False
            if hasattr(layer, "mask"):
                layer.mask = None

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y):
        dout = self.loss_fn.backward(y)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def parameters(self):
        """Yield the trainable parameters and current gradients."""
        return iter_trainable_params(self)

    def step(self, optimizer):
        """Update all trainable parameters with an external optimizer."""
        optimizer.step(self.parameters())

    def state_dict(self):
        """Return only the inference parameters, excluding runtime caches."""
        state = {"format_version": np.array(MODEL_FORMAT_VERSION, dtype=np.int64)}
        for index, layer in enumerate(self.layers):
            if hasattr(layer, "W"):
                state[f"layer_{index}_W"] = layer.W
                state[f"layer_{index}_b"] = layer.b
        return state

    def load_state_dict(self, state):
        """Load and validate an inference state produced by :meth:`state_dict`."""
        version = int(np.asarray(state["format_version"]).item())
        if version != MODEL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model format version {version}; expected {MODEL_FORMAT_VERSION}"
            )

        for index, layer in enumerate(self.layers):
            if not hasattr(layer, "W"):
                continue
            weight_key = f"layer_{index}_W"
            bias_key = f"layer_{index}_b"
            if weight_key not in state or bias_key not in state:
                raise ValueError(f"Model is missing parameters for layer {index}")

            weights = np.asarray(state[weight_key])
            bias = np.asarray(state[bias_key])
            if weights.shape != layer.W.shape or bias.shape != layer.b.shape:
                raise ValueError(
                    f"Shape mismatch for layer {index}: "
                    f"expected {layer.W.shape}/{layer.b.shape}, "
                    f"got {weights.shape}/{bias.shape}"
                )
            layer.W[...] = weights.astype(DTYPE, copy=False)
            layer.b[...] = bias.astype(DTYPE, copy=False)

        return self

    def save(self, filepath=DEFAULT_MODEL_PATH):
        """Save a compact, inference-only model state."""
        filepath = Path(filepath)
        if filepath.suffix.lower() != ".npz":
            raise ValueError("model files must use the .npz extension")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.eval()
        np.savez_compressed(filepath, **self.state_dict())
        print(f"Model saved to {filepath}")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_mnist_archive(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "mnist.npz"

    if archive.exists() and _sha256(archive) == MNIST_SHA256:
        return archive

    temporary = archive.with_suffix(".download")
    try:
        print(f"Downloading MNIST from {MNIST_URL}")
        urlretrieve(MNIST_URL, temporary)
        if _sha256(temporary) != MNIST_SHA256:
            raise RuntimeError("downloaded MNIST archive failed its SHA-256 check")
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    return archive


def load_mnist(data_dir=DEFAULT_DATA_DIR):
    """Load MNIST from a verified NumPy archive."""
    archive = _ensure_mnist_archive(data_dir)
    with np.load(archive, allow_pickle=False) as data:
        X_train = data["x_train"].astype(DTYPE)[:, None, :, :] / 255.0
        y_train = data["y_train"].astype(np.int64)
        X_test = data["x_test"].astype(DTYPE)[:, None, :, :] / 255.0
        y_test = data["y_test"].astype(np.int64)
    return X_train, y_train, X_test, y_test


def evaluate(model, X, y, batch_size=64):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    if len(X) == 0:
        raise ValueError("cannot evaluate an empty dataset")

    was_training = model.training
    model.eval()
    total_correct = 0
    total = 0

    try:
        for start in range(0, len(X), batch_size):
            X_batch = X[start : start + batch_size]
            y_batch = y[start : start + batch_size]

            logits = model.forward(X_batch)
            preds = np.argmax(logits, axis=1)

            total_correct += np.sum(preds == y_batch)
            total += len(y_batch)
    finally:
        if was_training:
            model.train()

    return total_correct / total


def start_training(
    epochs=3,
    batch_size=64,
    lr=5e-4,
    seed=42,
    data_dir=DEFAULT_DATA_DIR,
    model_path=DEFAULT_MODEL_PATH,
    train_limit=None,
    test_limit=None,
    optimizer_name="adam",
    dropout=True,
    weight_decay=0.0,
):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if train_limit is not None and train_limit <= 0:
        raise ValueError("train_limit must be positive")
    if test_limit is not None and test_limit <= 0:
        raise ValueError("test_limit must be positive")
    if not isinstance(optimizer_name, str) or not optimizer_name.strip():
        raise ValueError("optimizer_name must be a non-empty string")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    if seed is not None:
        np.random.seed(seed)

    X_train, y_train, X_test, y_test = load_mnist(data_dir)
    if train_limit is not None:
        X_train, y_train = X_train[:train_limit], y_train[:train_limit]
    if test_limit is not None:
        X_test, y_test = X_test[:test_limit], y_test[:test_limit]
    model = CNNModel(use_dropout=dropout)
    optimizer = build_optimizer(optimizer_name, lr, weight_decay=weight_decay)
    num_batches = (len(X_train) + batch_size - 1) // batch_size

    print(f"\nTraining with {optimizer_name}")

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        start_time = time.time()

        for batch_number, start in enumerate(range(0, len(X_train), batch_size)):
            batch_indices = indices[start : start + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            logits = model.forward(X_batch)
            loss = model.loss_fn.forward(logits, y_batch)

            model.backward(y_batch)
            model.step(optimizer)

            if batch_number % 50 == 0:
                acc = np.mean(np.argmax(logits, axis=1) == y_batch)
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch_number}/{num_batches} | "
                    f"Loss: {loss:.4f} | Acc: {acc:.4f}"
                )

        optimizer.lr *= 0.95
        print(
            f"Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds. "
            f"Next lr: {optimizer.lr:.6g}"
        )

        test_acc = evaluate(model, X_test, y_test, batch_size)
        print(f"Test Accuracy: {test_acc:.4f}")

    model.save(model_path)
    return model


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train the NumPy CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--optimizer",
        choices=optimizer_names(),
        default="adam",
        help="parameter update rule to use",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="decoupled weight decay for AdamW",
    )
    parser.add_argument(
        "--disable-dropout",
        action="store_true",
        help="disable dropout for controlled optimizer comparisons",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="train on only the first N samples (useful for smoke runs)",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="evaluate on only the first N test samples (useful for smoke runs)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_dir=args.data_dir,
        model_path=args.model,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        optimizer_name=args.optimizer,
        dropout=not args.disable_dropout,
        weight_decay=args.weight_decay,
    )
