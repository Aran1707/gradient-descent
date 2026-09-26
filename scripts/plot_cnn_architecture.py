"""Generate a slide-sized 2D diagram of the NumPy CNN architecture."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization.seminar_figures import _save, make_cnn_architecture


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 2D NumPy CNN architecture diagram."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "generated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _save(
        make_cnn_architecture(),
        args.output_dir / "numpy_cnn_architecture.png",
        args.output_dir / "numpy_cnn_architecture.svg",
    )
    print(f"Wrote CNN architecture figure to {args.output_dir}")


if __name__ == "__main__":
    main()
