"""Restrained optimizer animations for the seminar."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch

from landscapes.objectives import OBJECTIVES
from optimizers import build_optimizer

ANIMATION_SIZE = (8, 4.5)
ANIMATION_LRS = {
    "sgd": 0.18,
    "momentum": 0.03,
    "nesterov": 0.03,
    "adagrad": 0.30,
    "rmsprop": 0.08,
    "adam": 0.10,
    "adamw": 0.10,
    "lion": 0.04,
}
ANIMATION_LABELS = {
    "sgd": "Gradient descent",
    "momentum": "Momentum",
    "nesterov": "Nesterov",
    "adagrad": "AdaGrad",
    "rmsprop": "RMSProp",
    "adam": "Adam",
    "adamw": "AdamW",
    "lion": "Lion",
}
ANIMATION_OPTIMIZER_NAMES = (
    "sgd",
    "momentum",
    "nesterov",
    "adagrad",
    "rmsprop",
    "adam",
    "adamw",
    "lion",
)


def _trajectory(
    optimizer_name: str,
    objective_name: str,
    start: tuple[float, float],
    steps: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    objective, gradient_fn = OBJECTIVES[objective_name]
    theta = np.asarray(start, dtype=float)
    optimizer = build_optimizer(
        optimizer_name,
        ANIMATION_LRS[optimizer_name],
        weight_decay=0.01 if optimizer_name == "adamw" else 0.0,
    )
    points = [theta.copy()]
    gradients = []
    for _ in range(steps):
        gradient = np.asarray(gradient_fn(theta), dtype=float)
        gradients.append(gradient.copy())
        optimizer.step(((theta, gradient, "theta"),))
        points.append(theta.copy())
    del objective
    return np.asarray(points), gradients


def _arrow(ax, start, end, color: str, linestyle: str = "-") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=2.5,
            color=color,
            linestyle=linestyle,
            zorder=6,
        )
    )


def make_optimizer_animation(
    output: Path,
    optimizer_name: str = "sgd",
    objective_name: str = "sphere",
    start: tuple[float, float] = (2.6, 2.0),
    steps: int = 10,
    fps: int = 5,
) -> None:
    """Write one focused GIF showing one optimizer update at a time."""

    if optimizer_name == "sgd+momentum":
        optimizer_name = "momentum"
    if optimizer_name not in ANIMATION_OPTIMIZER_NAMES:
        raise ValueError(f"unsupported animation optimizer: {optimizer_name}")

    objective, gradient_fn = OBJECTIVES[objective_name]
    points, gradients = _trajectory(
        optimizer_name,
        objective_name,
        start,
        steps,
    )
    x_grid = np.linspace(-3.2, 3.2, 220)
    y_grid = np.linspace(-3.2, 3.2, 220)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    values = objective((x_mesh, y_mesh))

    frames = []
    for index in range(steps):
        frames.extend(
            (
                (index, "gradient"),
                (index, "update"),
                (index + 1, "arrive"),
            )
        )
    frames.append((steps, "finish"))

    fig, ax = plt.subplots(figsize=ANIMATION_SIZE, constrained_layout=True)

    def draw(frame):
        index, phase = frames[frame]
        ax.clear()
        ax.contour(
            x_mesh,
            y_mesh,
            values,
            levels=(2, 5, 10, 15, 25, 35),
            colors="#94a3b8",
            linewidths=0.9,
            alpha=0.6,
        )
        ax.plot(
            points[: index + 1, 0],
            points[: index + 1, 1],
            color="#2563eb",
            linewidth=3,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=4,
        )
        ax.scatter(
            [start[0]],
            [start[1]],
            marker="x",
            s=150,
            linewidths=3,
            color="#111827",
            zorder=5,
        )
        ax.scatter(
            [0],
            [0],
            marker="*",
            s=220,
            color="#dc2626",
            edgecolor="white",
            linewidth=1,
            zorder=5,
        )

        if phase == "gradient" and index < len(gradients):
            gradient = gradients[index]
            direction = gradient / max(np.linalg.norm(gradient), 1e-12)
            _arrow(
                ax,
                points[index],
                points[index] + 0.7 * direction,
                "#ea580c",
            )
            message = "gradient points uphill"
        elif phase == "update" and index < len(points) - 1:
            _arrow(
                ax,
                points[index],
                points[index + 1],
                "#2563eb",
            )
            message = r"$\theta \leftarrow \theta-\eta\nabla f$"
        elif phase == "finish":
            message = "same update, repeated"
        else:
            message = f"step {index}"

        ax.text(
            0.03,
            0.95,
            message,
            transform=ax.transAxes,
            fontsize=14,
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"},
        )
        ax.set(
            title=(
                f"{ANIMATION_LABELS[optimizer_name]} on "
                r"$f(x,y)=x^2+y^2$"
            ),
            xlabel="x",
            ylabel="y",
            xlim=(-3.2, 3.2),
            ylim=(-3.2, 3.2),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=11)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.grid(alpha=0.16)

    animation = FuncAnimation(
        fig,
        draw,
        frames=len(frames),
        interval=1000 // fps,
        repeat=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


def generate_optimizer_gifs(
    output_dir: Path,
    optimizer_names: tuple[str, ...],
    objective_name: str = "sphere",
) -> None:
    """Generate one GIF per selected optimizer using the shared visual style."""

    seen = set()
    for optimizer_name in optimizer_names:
        if optimizer_name == "sgd+momentum":
            optimizer_name = "momentum"
        if optimizer_name in seen:
            continue
        seen.add(optimizer_name)
        filename = (
            "gradient_descent.gif"
            if optimizer_name == "sgd"
            else f"{optimizer_name}.gif"
        )
        make_optimizer_animation(
            output_dir / filename,
            optimizer_name=optimizer_name,
            objective_name=objective_name,
        )


CNN_BLOCKS = (
    ("Input", "1×28×28", False),
    ("Conv + ReLU", "16×28×28", True),
    ("MaxPool", "16×14×14", True),
    ("Conv + ReLU", "32×14×14", True),
    ("MaxPool", "32×7×7", True),
    ("Flatten + Dense", "1568 → 128", False),
    ("Logits", "128 → 10", False),
)
CNN_STAGE_X = np.linspace(1.0, 13.0, len(CNN_BLOCKS))
CNN_NODE_YS = np.array([5.15, 4.55, 3.95, 3.35])


def _cnn_stage_nodes(x, abbreviated):
    """Return representative node locations and the omitted-node marker."""

    if abbreviated:
        return [
            np.array((x, CNN_NODE_YS[0])),
            np.array((x, CNN_NODE_YS[2])),
            np.array((x, CNN_NODE_YS[3])),
        ], CNN_NODE_YS[1]
    return [np.array((x, y)) for y in CNN_NODE_YS], None


def _cnn_connections(ax, source_nodes, target_nodes, masked_nodes=()):
    """Draw a sparse representative wiring pattern between two stages."""

    masked_nodes = tuple(masked_nodes)
    for source in source_nodes:
        for target in target_nodes:
            if abs(source[1] - target[1]) <= 0.9:
                is_masked = any(
                    np.array_equal(source, node) or np.array_equal(target, node)
                    for node in masked_nodes
                )
                ax.plot(
                    (source[0] + 0.15, target[0] - 0.15),
                    (source[1], target[1]),
                    color="#cbd5e1" if is_masked else "#94a3b8",
                    linewidth=0.9,
                    alpha=0.28 if is_masked else 0.65,
                    zorder=1,
                )


def _cnn_representative(nodes):
    """Select one stable node for the highlighted teaching path."""

    return min(nodes, key=lambda node: abs(node[1] - 4.55))


def _cnn_arrow(ax, start, end, color, linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=2.0,
            color=color,
            linestyle=linestyle,
        )
    )


def make_cnn_propagation_animation(
    output: Path,
    fps: int = 3,
) -> None:
    """Write a simple forward/backward flow GIF for the NumPy CNN."""

    frames = (
        [("forward", index) for index in range(len(CNN_BLOCKS))]
        + [("dropout", 5)]
        + [("loss", len(CNN_BLOCKS) - 1)]
        + [("backward", index) for index in range(len(CNN_BLOCKS) - 1, -1, -1)]
        + [("update", len(CNN_BLOCKS) - 1)]
    )
    node_sets = []
    for x_position, (_, _, abbreviated) in zip(CNN_STAGE_X, CNN_BLOCKS, strict=True):
        nodes, ellipsis_y = _cnn_stage_nodes(x_position, abbreviated)
        node_sets.append((nodes, ellipsis_y))

    fig, ax = plt.subplots(figsize=ANIMATION_SIZE, constrained_layout=True)

    def draw(frame):
        phase, index = frames[frame]
        ax.clear()
        ax.set_xlim(0, 14.6)
        ax.set_ylim(0, 7.2)
        ax.axis("off")
        ax.set_title(
            "NumPy CNN: forward pass, backpropagation, and update",
            fontsize=17,
            pad=8,
        )

        if phase == "forward":
            message = "forward pass: compute activations"
        elif phase == "loss":
            message = "loss: compare logits with the label"
        elif phase == "backward":
            message = "backpropagation: compute gradients"
        elif phase == "dropout":
            message = "dropout: one activation masked (training only)"
        else:
            message = "optimizer: update W and b"

        dropout_node = node_sets[5][0][2]
        masked_nodes = (dropout_node,) if phase == "dropout" else ()
        for block_index in range(len(CNN_BLOCKS) - 1):
            _cnn_connections(
                ax,
                node_sets[block_index][0],
                node_sets[block_index + 1][0],
                masked_nodes=masked_nodes,
            )

        for block_index, ((title, subtitle, _abbreviated), x_position) in enumerate(
            zip(CNN_BLOCKS, CNN_STAGE_X, strict=True)
        ):
            active = (
                (phase == "forward" and block_index <= index)
                or (phase == "backward" and block_index >= index)
                or phase == "dropout"
            )
            if phase == "loss" and block_index == len(CNN_BLOCKS) - 1:
                facecolor, edgecolor = "#fef3c7", "#b45309"
            elif phase == "update" and block_index in (1, 3, 5, 6):
                facecolor, edgecolor = "#dcfce7", "#15803d"
            elif active:
                facecolor, edgecolor = "#dbeafe", "#2563eb"
            else:
                facecolor, edgecolor = "#f8fafc", "#64748b"

            nodes, ellipsis_y = node_sets[block_index]
            for node in nodes:
                masked = phase == "dropout" and np.array_equal(node, dropout_node)
                node_facecolor = "#e5e7eb" if masked else facecolor
                node_edgecolor = "#9ca3af" if masked else edgecolor
                ax.add_patch(
                    Circle(
                        node,
                        radius=0.17,
                        facecolor=node_facecolor,
                        edgecolor=node_edgecolor,
                        linewidth=1.7,
                        zorder=3,
                    )
                )
                if masked:
                    ax.plot(
                        (node[0] - 0.11, node[0] + 0.11),
                        (node[1] - 0.11, node[1] + 0.11),
                        color="#6b7280",
                        linewidth=2.0,
                        zorder=4,
                    )
                    ax.plot(
                        (node[0] - 0.11, node[0] + 0.11),
                        (node[1] + 0.11, node[1] - 0.11),
                        color="#6b7280",
                        linewidth=2.0,
                        zorder=4,
                    )
            if ellipsis_y is not None:
                ax.text(
                    x_position,
                    ellipsis_y,
                    "⋮",
                    ha="center",
                    va="center",
                    fontsize=17,
                    color=edgecolor,
                    zorder=4,
                )
            ax.text(
                x_position,
                5.85,
                title,
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold",
            )
            ax.text(
                x_position,
                2.75,
                subtitle,
                ha="center",
                va="center",
                fontsize=8.5,
                color="#334155",
            )

        if phase == "forward" and index < len(CNN_BLOCKS) - 1:
            _cnn_arrow(
                ax,
                _cnn_representative(node_sets[index][0]) + np.array((0.18, 0.0)),
                _cnn_representative(node_sets[index + 1][0]) - np.array((0.18, 0.0)),
                "#2563eb",
            )
        elif phase == "backward" and index > 0:
            _cnn_arrow(
                ax,
                _cnn_representative(node_sets[index][0]) - np.array((0.18, 0.0)),
                _cnn_representative(node_sets[index - 1][0]) + np.array((0.18, 0.0)),
                "#ea580c",
            )
        elif phase == "update":
            _cnn_arrow(ax, (2.0, 6.35), (12.0, 6.35), "#15803d")

        ax.text(
            7.3,
            1.45,
            message,
            ha="center",
            va="center",
            fontsize=13.5,
            color={
                "forward": "#1d4ed8",
                "loss": "#b45309",
                "backward": "#c2410c",
                "dropout": "#6b7280",
                "update": "#15803d",
            }[phase],
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cbd5e1"},
        )

        ax.text(
            7.3,
            0.65,
            "backpropagation computes gradients; the selected optimizer owns the update rule",
            ha="center",
            va="center",
            fontsize=10.5,
            color="#475569",
        )

    animation = FuncAnimation(
        fig,
        draw,
        frames=len(frames),
        interval=1000 // fps,
        repeat=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
