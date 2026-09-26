"""Generate focused optimizer GIFs for the seminar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from optimizers import optimizer_names
from visualization.animations import (
    ANIMATION_OPTIMIZER_NAMES,
    generate_optimizer_gifs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one restrained GIF per selected optimizer."
    )
    parser.add_argument(
        "--optimizer",
        choices=optimizer_names(),
        action="append",
        dest="optimizers",
        help="optimizer to animate; repeat this option to select several",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="generate the main optimizer GIF for every registered optimizer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures" / "gif",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all and args.optimizers:
        raise ValueError("use --all or --optimizer, not both")
    selected = (
        tuple(ANIMATION_OPTIMIZER_NAMES)
        if args.all
        else tuple(args.optimizers or ("sgd",))
    )
    generate_optimizer_gifs(args.output_dir, selected)
    print(f"Wrote optimizer GIFs to {args.output_dir}")


if __name__ == "__main__":
    main()
