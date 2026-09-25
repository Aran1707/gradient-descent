"""Reproducible 2D optimizer trajectory generator and CSV exporter.

Exports deterministic optimization trajectories for classroom demonstration,
testing, and offline analysis according to PROJECT_PLAN.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from landscapes.objectives import OBJECTIVES
from optimizers import build_optimizer


TRAJECTORY_CONFIGS: dict[str, dict[str, Any]] = {
    "gd": {
        "optimizer": "sgd",
        "lr": 0.03,
        "steps": 36,
        "weight_decay": 0.0,
        "filename": "traj_gd.csv",
    },
    "momentum": {
        "optimizer": "momentum",
        "lr": 0.006,
        "steps": 55,
        "weight_decay": 0.0,
        "filename": "traj_momentum.csv",
    },
    "nesterov": {
        "optimizer": "nesterov",
        "lr": 0.008,
        "steps": 55,
        "weight_decay": 0.0,
        "filename": "traj_nesterov.csv",
    },
    "adagrad": {
        "optimizer": "adagrad",
        "lr": 0.30,
        "steps": 45,
        "weight_decay": 0.0,
        "filename": "traj_adagrad.csv",
    },
    "rmsprop": {
        "optimizer": "rmsprop",
        "lr": 0.07,
        "steps": 45,
        "weight_decay": 0.0,
        "filename": "traj_rmsprop.csv",
    },
    "adam": {
        "optimizer": "adam",
        "lr": 0.10,
        "steps": 45,
        "weight_decay": 0.0,
        "filename": "traj_adam.csv",
    },
    "adamw": {
        "optimizer": "adamw",
        "lr": 0.10,
        "steps": 45,
        "weight_decay": 0.02,
        "filename": "traj_adamw.csv",
    },
    "lion": {
        "optimizer": "lion",
        "lr": 0.04,
        "steps": 55,
        "weight_decay": 0.0,
        "filename": "traj_lion.csv",
    },
}


def export_trajectory(
    opt_key: str,
    cfg: dict[str, Any],
    objective_name: str,
    start: tuple[float, float],
    output_dir: Path,
) -> dict[str, Any]:
    """Simulate trajectory using the shared optimizer implementation and save to CSV."""
    import numpy as np

    objective_fn, grad_fn = OBJECTIVES[objective_name]
    theta = np.asarray(start, dtype=float).copy()

    optimizer = build_optimizer(
        cfg["optimizer"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    records: list[tuple[int, float, float, float]] = []
    loss = float(objective_fn(theta))
    records.append((0, float(theta[0]), float(theta[1]), loss))

    status = "completed"
    for step in range(1, cfg["steps"] + 1):
        grad = np.asarray(grad_fn(theta), dtype=float)
        optimizer.step(((theta, grad, "theta"),))
        loss = float(objective_fn(theta))
        records.append((step, float(theta[0]), float(theta[1]), loss))
        if not np.all(np.isfinite(theta)):
            status = "diverged"
            break

    csv_path = output_dir / cfg["filename"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "loss"])
        for row in records:
            writer.writerow(row)

    return {
        "key": opt_key,
        "optimizer": cfg["optimizer"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "steps_budget": cfg["steps"],
        "steps_recorded": len(records) - 1,
        "objective": objective_name,
        "start": list(start),
        "final_point": [float(theta[0]), float(theta[1])],
        "final_loss": loss,
        "status": status,
        "file": str(csv_path.name),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export reproducible 2D optimization trajectories to CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory to write trajectory CSVs and manifest into.",
    )
    parser.add_argument(
        "--objective",
        choices=list(OBJECTIVES.keys()),
        default="ill-conditioned",
        help="Target 2D objective function.",
    )
    parser.add_argument(
        "--start-x",
        type=float,
        default=2.8,
        help="Initial x coordinate.",
    )
    parser.add_argument(
        "--start-y",
        type=float,
        default=1.0,
        help="Initial y coordinate.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        choices=list(TRAJECTORY_CONFIGS.keys()),
        default=None,
        help="Subset of optimizers to export (default: all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start = (args.start_x, args.start_y)

    keys = args.optimizers or list(TRAJECTORY_CONFIGS.keys())
    manifest = {
        "objective": args.objective,
        "start": list(start),
        "trajectories": [],
    }

    print(f"Exporting trajectories for objective {args.objective!r} starting at {start}...")
    for key in keys:
        cfg = TRAJECTORY_CONFIGS[key]
        meta = export_trajectory(key, cfg, args.objective, start, output_dir)
        manifest["trajectories"].append(meta)
        print(f"  [{key:<9}] -> {meta['file']:<18} | final: ({meta['final_point'][0]:.4f}, {meta['final_point'][1]:.4f}) | loss: {meta['final_loss']:.4e} | status: {meta['status']}")

    manifest_path = output_dir / "trajectories_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {len(keys)} trajectory CSV files and manifest to {output_dir}")


if __name__ == "__main__":
    main()
