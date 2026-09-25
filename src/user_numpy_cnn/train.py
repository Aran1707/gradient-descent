import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optimizers import SAM, build_optimizer, iter_trainable_params, optimizer_names

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
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)  # pyright: ignore[reportArgumentType]

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
        np.savez_compressed(filepath, **self.state_dict())  # pyright: ignore[reportArgumentType]
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


def evaluate_loss_and_acc(model, X, y, batch_size=64):
    """Compute average loss and classification accuracy."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    if len(X) == 0:
        raise ValueError("cannot evaluate an empty dataset")

    was_training = model.training
    model.eval()
    total_correct = 0
    total_loss = 0.0
    total = 0

    try:
        for start in range(0, len(X), batch_size):
            X_batch = X[start : start + batch_size]
            y_batch = y[start : start + batch_size]

            logits = model.forward(X_batch)
            loss = model.loss_fn.forward(logits, y_batch)
            preds = np.argmax(logits, axis=1)

            total_loss += float(loss) * len(y_batch)
            total_correct += int(np.sum(preds == y_batch))
            total += len(y_batch)
    finally:
        if was_training:
            model.train()

    return total_loss / total, total_correct / total


def evaluate(model, X, y, batch_size=64):
    """Compute classification accuracy (preserved for backwards compatibility)."""
    _, acc = evaluate_loss_and_acc(model, X, y, batch_size)
    return acc


def start_training(
    epochs=3,
    batch_size=64,
    lr=5e-4,
    seed=42,
    data_dir=DEFAULT_DATA_DIR,
    model_path=DEFAULT_MODEL_PATH,
    train_limit=None,
    test_limit=None,
    val_size=2000,
    optimizer_name="adam",
    dropout=True,
    weight_decay=0.0,
    lr_decay=0.95,
    rho=0.05,
    metrics_path=None,
    save_model=True,
):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if val_size is not None and val_size < 0:
        raise ValueError("val_size must be non-negative")
    if train_limit is not None and train_limit <= 0:
        raise ValueError("train_limit must be positive")
    if test_limit is not None and test_limit <= 0:
        raise ValueError("test_limit must be positive")
    if not isinstance(optimizer_name, str) or not optimizer_name.strip():
        raise ValueError("optimizer_name must be a non-empty string")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if lr_decay <= 0 or lr_decay > 1.0:
        raise ValueError("lr_decay must be in (0, 1]")

    if seed is not None:
        np.random.seed(seed)

    X_train_raw, y_train_raw, X_test_raw, y_test_raw = load_mnist(data_dir)

    # Separate training data into training and validation splits
    if val_size and val_size > 0 and len(X_train_raw) > val_size:
        X_val = X_train_raw[-val_size:]
        y_val = y_train_raw[-val_size:]
        X_pool = X_train_raw[:-val_size]
        y_pool = y_train_raw[:-val_size]
    else:
        X_val, y_val = None, None
        X_pool = X_train_raw
        y_pool = y_train_raw

    if train_limit is not None:
        X_train, y_train = X_pool[:train_limit], y_pool[:train_limit]
    else:
        X_train, y_train = X_pool, y_pool

    if test_limit is not None:
        X_test, y_test = X_test_raw[:test_limit], y_test_raw[:test_limit]
    else:
        X_test, y_test = X_test_raw, y_test_raw

    model = CNNModel(use_dropout=dropout)
    optimizer = build_optimizer(optimizer_name, lr, weight_decay=weight_decay, rho=rho)
    is_sam = isinstance(optimizer, SAM)
    num_batches = (len(X_train) + batch_size - 1) // batch_size

    print(
        f"\nTraining with optimizer: {optimizer_name} | "
        f"train samples: {len(X_train)} | val samples: {len(X_val) if X_val is not None else 0} | "
        f"test samples: {len(X_test)} | seed: {seed}"
    )

    history = {
        "optimizer": optimizer_name,
        "lr_initial": lr,
        "lr_decay": lr_decay,
        "weight_decay": weight_decay,
        "rho": rho if is_sam else None,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": len(X_train),
        "val_samples": len(X_val) if X_val is not None else 0,
        "test_samples": len(X_test),
        "epoch_metrics": [],
    }

    start_total_time = time.time()

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        epoch_start_time = time.time()
        running_loss = 0.0
        running_correct = 0
        samples_seen = 0

        for batch_number, start in enumerate(range(0, len(X_train), batch_size)):
            batch_indices = indices[start : start + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            if is_sam:
                # SAM step 1: forward & backward at current w
                logits = model.forward(X_batch)
                loss = model.loss_fn.forward(logits, y_batch)
                model.backward(y_batch)
                optimizer.first_step(model.parameters())

                # SAM step 2: forward & backward at perturbed w + e
                logits_perturbed = model.forward(X_batch)
                _ = model.loss_fn.forward(logits_perturbed, y_batch)
                model.backward(y_batch)
                optimizer.second_step(model.parameters())
            else:
                logits = model.forward(X_batch)
                loss = model.loss_fn.forward(logits, y_batch)
                model.backward(y_batch)
                model.step(optimizer)

            batch_correct = int(np.sum(np.argmax(logits, axis=1) == y_batch))
            running_loss += float(loss) * len(y_batch)
            running_correct += batch_correct
            samples_seen += len(y_batch)

            if batch_number % 50 == 0:
                acc = batch_correct / len(y_batch)
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {batch_number}/{num_batches} | "
                    f"Loss: {loss:.4f} | Acc: {acc:.4f}"
                )

        train_loss = running_loss / samples_seen
        train_acc = running_correct / samples_seen
        epoch_duration = time.time() - epoch_start_time

        # Update learning rate if decay is specified
        current_lr = optimizer.lr
        if lr_decay != 1.0:
            optimizer.lr *= lr_decay

        metric_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "lr": current_lr,
            "duration_sec": epoch_duration,
        }

        # Evaluate on validation split after each epoch
        if X_val is not None and len(X_val) > 0:
            val_loss, val_acc = evaluate_loss_and_acc(model, X_val, y_val, batch_size)
            metric_entry["val_loss"] = val_loss
            metric_entry["val_acc"] = val_acc
            print(
                f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
            )

        history["epoch_metrics"].append(metric_entry)

    # Scientific evaluation: evaluate official test set only ONCE at the end
    test_loss, test_acc = evaluate_loss_and_acc(model, X_test, y_test, batch_size)
    total_time = time.time() - start_total_time
    history["final_test_loss"] = test_loss
    history["final_test_acc"] = test_acc
    history["total_duration_sec"] = total_time

    print(
        f"\nFinal Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.4f} | "
        f"Total training time: {total_time:.2f}s"
    )

    if save_model:
        model.save(model_path)

    if metrics_path is not None:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with metrics_file.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Metrics saved to {metrics_file}")

    return model, history


def run_multi_seed_experiments(
    seeds=(0, 1, 2),
    model_path=DEFAULT_MODEL_PATH,
    metrics_path=None,
    **kwargs,
):
    """Run controlled experiment protocol across multiple seeds and report statistics.

    Saves distinct per-seed model artifacts (e.g. cnn-model_seed_0.npz) and per-seed
    metrics files (e.g. metrics_seed_0.json), followed by an aggregate summary.
    """
    results = []
    base_model_path = Path(model_path) if model_path is not None else None
    base_metrics_path = Path(metrics_path) if metrics_path is not None else None

    for s in seeds:
        print(f"\n{'=' * 20} Running Seed {s} {'=' * 20}")
        seed_model_path = (
            base_model_path.with_name(f"{base_model_path.stem}_seed_{s}{base_model_path.suffix}")
            if base_model_path is not None
            else None
        )
        seed_metrics_path = (
            base_metrics_path.with_name(f"{base_metrics_path.stem}_seed_{s}{base_metrics_path.suffix}")
            if base_metrics_path is not None
            else None
        )

        _, hist = start_training(
            seed=s,
            model_path=seed_model_path or DEFAULT_MODEL_PATH,
            save_model=(seed_model_path is not None),
            metrics_path=seed_metrics_path,
            **kwargs,
        )
        results.append(hist)

    test_accs = [r["final_test_acc"] for r in results]
    mean_acc = float(np.mean(test_accs))
    std_acc = float(np.std(test_accs))

    test_losses = [r["final_test_loss"] for r in results]
    mean_loss = float(np.mean(test_losses))
    std_loss = float(np.std(test_losses))

    print(f"\n{'=' * 20} Multi-Seed Experiment Summary ({len(seeds)} seeds) {'=' * 20}")
    print(f"Optimizer: {results[0]['optimizer']}")
    for idx, (s, acc) in enumerate(zip(seeds, test_accs, strict=True)):
        print(f"  Seed {s}: Test Accuracy = {acc:.4f} | Test Loss = {test_losses[idx]:.4f}")
    print(f"Mean Test Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"Mean Test Loss: {mean_loss:.4f} +/- {std_loss:.4f}")

    summary = {
        "optimizer": results[0]["optimizer"],
        "seeds": list(seeds),
        "mean_test_acc": mean_acc,
        "std_test_acc": std_acc,
        "mean_test_loss": mean_loss,
        "std_test_loss": std_loss,
        "results": results,
    }

    if base_metrics_path is not None:
        base_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with base_metrics_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Aggregate multi-seed metrics saved to {base_metrics_path}")

    return summary


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train the NumPy CNN on MNIST.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.95,
        help="learning rate multiplicative decay factor per epoch (default: 0.95)",
    )
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
        "--rho",
        type=float,
        default=0.05,
        help="neighborhood perturbation radius for SAM",
    )
    parser.add_argument(
        "--disable-dropout",
        action="store_true",
        help="disable dropout for controlled optimizer comparisons",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated list of seeds for multi-seed evaluation (e.g. '0,1,2')",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="train on only the first N samples (useful for smoke runs)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=2000,
        help="number of validation samples split from training set (default: 2000)",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="evaluate on only the first N test samples (useful for smoke runs)",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="optional path to write JSON run metrics and history",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seeds is not None:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        run_multi_seed_experiments(
            seeds=seed_list,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay=args.lr_decay,
            data_dir=args.data_dir,
            model_path=args.model,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            val_size=args.val_size,
            optimizer_name=args.optimizer,
            dropout=not args.disable_dropout,
            weight_decay=args.weight_decay,
            rho=args.rho,
            metrics_path=args.metrics_file,
        )
    else:
        start_training(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay=args.lr_decay,
            seed=args.seed,
            data_dir=args.data_dir,
            model_path=args.model,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            val_size=args.val_size,
            optimizer_name=args.optimizer,
            dropout=not args.disable_dropout,
            weight_decay=args.weight_decay,
            rho=args.rho,
            metrics_path=args.metrics_file,
        )
