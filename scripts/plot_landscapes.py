"""Generate the seminar's gradient-field and 3D surface figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visualization.seminar_figures import _save, make_gradient_field, make_surface


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 2D gradient field and 3D objective surface."
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
        make_gradient_field(),
        args.output_dir / "sphere_gradient_field.png",
        args.output_dir / "sphere_gradient_field.svg",
    )
    _save(
        make_surface(),
        args.output_dir / "sphere_surface_3d.png",
        args.output_dir / "sphere_surface_3d.svg",
    )
    print(f"Wrote landscape figures to {args.output_dir}")


if __name__ == "__main__":
    main()
