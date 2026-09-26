"""Generate the single-landscape optimizer comparison figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization.seminar_figures import _save, make_optimizer_comparison


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare optimizer trajectories on one shared landscape."
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
        make_optimizer_comparison(),
        args.output_dir / "optimizer_trajectories.png",
        args.output_dir / "optimizer_trajectories.svg",
    )
    print(f"Wrote optimizer trajectory figure to {args.output_dir}")


if __name__ == "__main__":
    main()
