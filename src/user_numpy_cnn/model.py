import argparse
import pickle
import warnings
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from train import CNNModel, DEFAULT_MODEL_PATH, MODEL_FORMAT_VERSION, PROJECT_DIR


class RedirectUnpickler(pickle.Unpickler):
    """Read model files produced when the old training script was run directly."""

    def find_class(self, module, name):
        if module == "__main__":
            module = "train"
        return super().find_class(module, name)


def preprocess_image(image_path, invert=None, preserve_aspect=True):
    """Convert an image into a centered MNIST-shaped float32 batch."""
    with Image.open(image_path) as source:
        image = source.convert("L")

    if invert is None:
        # MNIST uses a dark background; the bundled hand-written examples use
        # a light background. Auto-detect that polarity unless it is explicit.
        invert = np.asarray(image, dtype=np.float32).mean() > 127.0
    if invert:
        image = ImageOps.invert(image)

    if preserve_aspect:
        image.thumbnail((28, 28), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (28, 28), color=0)
        offset = ((28 - image.width) // 2, (28 - image.height) // 2)
        canvas.paste(image, offset)
        image = canvas
    else:
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return pixels.reshape(1, 1, 28, 28)


class DigitRecognizer:
    def __init__(
        self, model_path=DEFAULT_MODEL_PATH, invert=None, preserve_aspect=True
    ):
        self.model_path = Path(model_path)
        self.invert = invert
        self.preserve_aspect = preserve_aspect
        self.model = self._load_model(self.model_path)
        self.layers = self.model.layers

    @staticmethod
    def _load_model(model_path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Could not find model file: {model_path}. "
                "Run train.py first or pass --model with a valid file."
            )

        if model_path.suffix.lower() == ".npz":
            try:
                with np.load(model_path, allow_pickle=False) as state:
                    version = int(np.asarray(state["format_version"]).item())
                    if version != MODEL_FORMAT_VERSION:
                        raise ValueError(
                            f"Unsupported model format version {version}; "
                            f"expected {MODEL_FORMAT_VERSION}"
                        )
                    model = CNNModel().load_state_dict(state)
            except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
                raise ValueError(f"Invalid model file {model_path}: {exc}") from exc
            model.eval()
            return model

        if model_path.suffix.lower() == ".pkl":
            warnings.warn(
                "Loading pickle model files executes code contained in the file; "
                "use a trusted legacy model only. Re-save it as .npz afterward.",
                RuntimeWarning,
                stacklevel=2,
            )
            with model_path.open("rb") as file:
                layers = RedirectUnpickler(file).load()
            model = CNNModel()
            model.layers = layers
            model.eval()
            return model

        raise ValueError("model files must use the .npz or legacy .pkl extension")

    def __call__(self, image_path):
        """Return the predicted digit for an image file."""
        inputs = preprocess_image(
            image_path,
            invert=self.invert,
            preserve_aspect=self.preserve_aspect,
        )
        logits = inputs
        for layer in self.layers:
            logits = layer.forward(logits)
        return int(np.argmax(logits, axis=1)[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a digit with the NumPy CNN.")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=PROJECT_DIR / "test-assets" / "test.png",
        help="image to classify (default: test-assets/test.png)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    polarity = parser.add_mutually_exclusive_group()
    polarity.add_argument(
        "--invert",
        dest="invert",
        action="store_true",
        help="invert a light-background image",
    )
    polarity.add_argument(
        "--no-invert",
        dest="invert",
        action="store_false",
        help="use an image that already has a white digit on a dark background",
    )
    parser.set_defaults(invert=None)
    parser.add_argument(
        "--stretch",
        action="store_true",
        help="resize directly to 28x28 instead of preserving the aspect ratio",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    recognizer = DigitRecognizer(
        args.model,
        invert=args.invert,
        preserve_aspect=not args.stretch,
    )
    print(f"Predicted digit: {recognizer(args.image)}")


if __name__ == "__main__":
    main()
