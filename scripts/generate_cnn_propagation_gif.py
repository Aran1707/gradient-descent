"""Generate the focused NumPy CNN propagation animation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization.animations import make_cnn_propagation_animation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a forward/backward NumPy CNN propagation GIF."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "figures" / "gif" / "cnn_forward_backward.gif",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_cnn_propagation_animation(args.output)
    print(f"Wrote CNN propagation GIF to {args.output}")


if __name__ == "__main__":
    main()
