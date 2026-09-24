"""Regenerate all ten prepared PNG/SVG seminar figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization.seminar_figures import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate all prepared gradient-descent seminar figures."
    )
    parser.add_argument(
        "--png-dir",
        type=Path,
        default=ROOT / "figures" / "png",
    )
    parser.add_argument(
        "--svg-dir",
        type=Path,
        default=ROOT / "figures" / "svg",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_all_figures(args.png_dir, args.svg_dir)
    print(f"Wrote PNG figures to {args.png_dir}")
    print(f"Wrote SVG figures to {args.svg_dir}")


if __name__ == "__main__":
    main()
