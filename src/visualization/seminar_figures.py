"""Readable, slide-sized figures for the gradient-descent seminar.

The functions in this module deliberately favor one visible teaching point per
figure over dense dashboards. PNG and SVG outputs are generated together so
the prepared assets remain reproducible and editable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from landscapes.objectives import OBJECTIVES, grad_double_well
from optimizers import build_optimizer

FIGURE_SIZE = (12.8, 7.2)
TITLE_SIZE = 24
LABEL_SIZE = 17
TICK_SIZE = 13
LEGEND_SIZE = 13

OPTIMIZER_COLORS = {
    "sgd": "#2563eb",
    "momentum": "#ea580c",
    "nesterov": "#0891b2",
    "adagrad": "#16a34a",
    "rmsprop": "#9333ea",
    "adam": "#dc2626",
    "adamw": "#c026d3",
    "lion": "#ca8a04",
}

OPTIMIZER_LABELS = {
    "sgd": "SGD",
    "momentum": "Momentum",
    "nesterov": "Nesterov",
    "adagrad": "AdaGrad",
    "rmsprop": "RMSProp",
    "adam": "Adam",
    "adamw": "AdamW",
    "lion": "Lion",
}

OPTIMIZER_SETTINGS = {
    "sgd": (0.03, 36, 0.0),
    "momentum": (0.006, 55, 0.0),
    "nesterov": (0.008, 55, 0.0),
    "adagrad": (0.30, 45, 0.0),
    "rmsprop": (0.07, 45, 0.0),
    "adam": (0.10, 45, 0.0),
    "adamw": (0.10, 45, 0.02),
    "lion": (0.04, 55, 0.0),
}


def _style_axes(ax, *, equal: bool = False) -> None:
    ax.tick_params(labelsize=TICK_SIZE)
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.grid(alpha=0.18)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def _save(fig, png_path: Path, svg_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        svg_path,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _objective_grid(
    objective_name: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    resolution: int = 240,
):
    objective, _ = OBJECTIVES[objective_name]
    x_values = np.linspace(*x_limits, resolution)
    y_values = np.linspace(*y_limits, resolution)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    return x_grid, y_grid, objective((x_grid, y_grid))


def _contours(
    ax,
    objective_name: str,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    levels: Iterable[float],
):
    x_grid, y_grid, values = _objective_grid(
        objective_name,
        x_limits,
        y_limits,
    )
    contour = ax.contour(
        x_grid,
        y_grid,
        values,
        levels=list(levels),
        colors="#64748b",
        linewidths=1.0,
        alpha=0.55,
    )
    return x_grid, y_grid, values, contour


def _gradient_descent(
    start: tuple[float, float],
    learning_rate: float,
    gradient_fn: Callable,
    steps: int,
) -> np.ndarray:
    theta = np.asarray(start, dtype=float)
    trajectory = [theta.copy()]
    for _ in range(steps):
        theta = theta - learning_rate * np.asarray(gradient_fn(theta))
        trajectory.append(theta.copy())
    return np.asarray(trajectory)


def optimizer_trajectory(
    name: str,
    learning_rate: float,
    start: tuple[float, float],
    gradient_fn: Callable,
    steps: int,
    weight_decay: float = 0.0,
) -> np.ndarray:
    theta = np.asarray(start, dtype=float)
    optimizer = build_optimizer(name, learning_rate, weight_decay=weight_decay)
    trajectory = [theta.copy()]
    for _ in range(steps):
        if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > 1e10):
            break
        gradient = np.asarray(gradient_fn(theta), dtype=float)
        optimizer.step(((theta, gradient, "theta"),))
        trajectory.append(theta.copy())
        if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > 1e10):
            break
    return np.asarray(trajectory)


def make_gradient_field(
    objective_name: str = "sphere",
    x_limits: tuple[float, float] = (-3.0, 3.0),
    y_limits: tuple[float, float] = (-3.0, 3.0),
):
    """Build a sparse, direction-first gradient-field figure."""

    _, gradient_fn = OBJECTIVES[objective_name]
    coordinates_x = np.linspace(*x_limits, 13)
    coordinates_y = np.linspace(*y_limits, 13)
    x_grid, y_grid = np.meshgrid(coordinates_x, coordinates_y)
    grad_x, grad_y = gradient_fn((x_grid, y_grid))
    magnitude = np.hypot(grad_x, grad_y)
    safe_magnitude = np.where(magnitude == 0, 1.0, magnitude)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    _contours(
        ax,
        objective_name,
        x_limits,
        y_limits,
        levels=(2, 5, 8, 12, 16),
    )
    # The contours above are only intended as quiet level lines. The arrows
    # carry the teaching message and are normalized so every direction stays
    # legible at slide scale.
    ax.quiver(
        x_grid,
        y_grid,
        grad_x / safe_magnitude,
        grad_y / safe_magnitude,
        color="#1d4ed8",
        angles="xy",
        scale_units="xy",
        scale=2.6,
        width=0.004,
        headwidth=4.5,
        headlength=5.5,
    )
    ax.scatter(
        [0],
        [0],
        marker="*",
        s=260,
        color="#dc2626",
        edgecolor="white",
        linewidth=1.0,
        zorder=4,
    )
    ax.annotate(
        "minimum",
        xy=(0, 0),
        xytext=(0.55, -0.55),
        fontsize=LABEL_SIZE,
        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.5},
        color="#991b1b",
    )
    ax.set(
        title=r"Gradient field for $f(x,y)=x^2+y^2$",
        xlabel="x",
        ylabel="y",
        xlim=x_limits,
        ylim=y_limits,
    )
    _style_axes(ax, equal=True)
    return fig


def make_surface(
    objective_name: str = "sphere",
    x_limits: tuple[float, float] = (-3.0, 3.0),
    y_limits: tuple[float, float] = (-3.0, 3.0),
):
    """Build a simple, uncluttered 3D objective surface."""

    objective, _ = OBJECTIVES[objective_name]
    x_grid, y_grid, values = _objective_grid(
        objective_name,
        x_limits,
        y_limits,
        resolution=90,
    )
    fig = plt.figure(figsize=FIGURE_SIZE, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        values,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    ax.scatter([0], [0], zs=int(objective((0, 0))), color="#dc2626", s=90)
    ax.set(
        title=r"Loss surface: $z=x^2+y^2$",
        xlabel="x",
        ylabel="y",
    )
    ax.tick_params(labelsize=TICK_SIZE)
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.zaxis.label.set_fontsize(LABEL_SIZE)
    ax.view_init(elev=28, azim=-55)
    ax.set_box_aspect((1, 1, 0.8))
    # A colorbar adds a second scale without helping the first explanation.
    del surface
    return fig


def make_gd_steps():
    """Build the first trajectory figure with only the important points."""

    trajectory = _gradient_descent(
        (2.6, 2.0),
        learning_rate=0.18,
        gradient_fn=lambda theta: 2 * np.asarray(theta),
        steps=8,
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    _contours(
        ax,
        "sphere",
        (-3.2, 3.2),
        (-3.2, 3.2),
        levels=(2, 5, 10, 15, 25, 35),
    )
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="#111827",
        linewidth=3,
        marker="o",
        markersize=8,
        markerfacecolor="#2563eb",
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )
    ax.scatter(
        [trajectory[0, 0]],
        [trajectory[0, 1]],
        marker="x",
        s=170,
        linewidths=3,
        color="#111827",
        zorder=4,
    )
    ax.scatter([0], [0], marker="*", s=300, color="#dc2626", zorder=4)
    ax.annotate(
        "start",
        xy=trajectory[0],
        xytext=(2.25, 2.65),
        fontsize=LABEL_SIZE,
        arrowprops={"arrowstyle": "->", "lw": 1.5},
    )
    ax.annotate(
        "minimum",
        xy=(0, 0),
        xytext=(-1.6, -0.8),
        fontsize=LABEL_SIZE,
        color="#991b1b",
        arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.5},
    )
    ax.set(
        title=r"Gradient descent on $f(x,y)=x^2+y^2$",
        xlabel="x",
        ylabel="y",
        xlim=(-3.2, 3.2),
        ylim=(-3.2, 3.2),
    )
    _style_axes(ax, equal=True)
    return fig


def make_learning_rate():
    """Compare three learning-rate regimes on one shared landscape."""

    start = (2.8, 1.0)
    x_limits = (-3.2, 3.2)
    y_limits = (-2.5, 2.5)
    rates = (
        (0.005, "small: slow", "#2563eb", 35),
        (0.03, "good: converges", "#16a34a", 22),
        (0.05, "too large: diverges", "#dc2626", 5),
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    _contours(
        ax,
        "ill-conditioned",
        x_limits,
        y_limits,
        levels=(2, 5, 10, 20, 35, 50, 75, 100),
    )
    for rate, label, color, steps in rates:
        trajectory = _gradient_descent(
            start,
            rate,
            lambda theta: np.array([2 * theta[0], 50 * theta[1]]),
            steps,
        )
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linewidth=3,
            marker="o",
            markersize=6,
            markevery=max(1, len(trajectory) // 7),
            label=rf"$\eta={rate:g}$ — {label}",
        )
    ax.scatter(*start, marker="x", s=180, linewidths=3, color="#111827", zorder=4)
    ax.scatter([0], [0], marker="*", s=280, color="#111827", zorder=4)
    ax.annotate(
        r"stability boundary: $\eta<0.04$",
        xy=(0, 0),
        xytext=(-2.9, -2.05),
        fontsize=LABEL_SIZE,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#94a3b8"},
    )
    ax.set(
        title=r"Learning rate on $f(x,y)=x^2+25y^2$",
        xlabel="x",
        ylabel="y",
        xlim=x_limits,
        ylim=y_limits,
    )
    _style_axes(ax, equal=True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=LEGEND_SIZE,
        frameon=False,
    )
    return fig


def make_optimizer_comparison(
    objective_name: str = "ill-conditioned",
    optimizer_names: tuple[str, ...] = tuple(OPTIMIZER_SETTINGS),
):
    """Compare optimizer paths on one large, shared contour landscape."""

    start = (2.8, 1.0)
    x_limits = (-3.2, 3.2)
    y_limits = (-1.25, 1.25)
    gradient_fn = OBJECTIVES[objective_name][1]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    _contours(
        ax,
        objective_name,
        x_limits,
        y_limits,
        levels=(2, 5, 10, 18, 28, 40, 55),
    )
    handles = []
    labels = []
    for name in optimizer_names:
        learning_rate, steps, weight_decay = OPTIMIZER_SETTINGS[name]
        trajectory = optimizer_trajectory(
            name,
            learning_rate,
            start,
            gradient_fn,
            steps,
            weight_decay=weight_decay,
        )
        (line,) = ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=OPTIMIZER_COLORS[name],
            linewidth=2.8,
            marker="o",
            markersize=4.5,
            markevery=max(1, len(trajectory) // 9),
            alpha=0.94,
        )
        handles.append(line)
        suffix = rf" ($\eta={learning_rate:g}$)"
        labels.append(OPTIMIZER_LABELS[name] + suffix)

    ax.scatter(*start, marker="x", s=200, linewidths=3, color="#111827", zorder=5)
    ax.scatter(
        [0],
        [0],
        marker="*",
        s=300,
        color="#111827",
        edgecolor="white",
        linewidth=1,
        zorder=5,
    )
    ax.annotate(
        "same start",
        xy=start,
        xytext=(1.65, 1.08),
        fontsize=LABEL_SIZE,
        arrowprops={"arrowstyle": "->", "lw": 1.5},
    )
    ax.annotate(
        "global minimum",
        xy=(0, 0),
        xytext=(-1.3, -0.95),
        fontsize=LABEL_SIZE,
        arrowprops={"arrowstyle": "->", "lw": 1.5},
    )
    ax.set(
        title=r"Optimizer paths on $f(x,y)=x^2+25y^2$",
        xlabel="x",
        ylabel="y",
        xlim=x_limits,
        ylim=y_limits,
    )
    _style_axes(ax, equal=True)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        fontsize=LEGEND_SIZE,
        frameon=False,
        columnspacing=1.2,
    )
    return fig


def make_nonconvex():
    """Show two basins only: one local minimum and the lower global minimum."""

    x_limits = (-1.8, 1.8)
    y_limits = (-1.25, 1.25)
    minima = (-1.0241203, 0.97399435)
    trajectories = (
        ((-1.35, 0.85), "#2563eb", "global basin"),
        ((1.35, 0.85), "#dc2626", "local basin"),
    )

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    x_grid, y_grid, values = _objective_grid(
        "double-well",
        x_limits,
        y_limits,
    )
    contour = ax.contour(
        x_grid,
        y_grid,
        values,
        levels=(-0.15, 0, 0.2, 0.5, 1, 2, 3, 4),
        colors="#64748b",
        linewidths=1.1,
        alpha=0.6,
    )
    del contour
    for start, color, label in trajectories:
        path = _gradient_descent(
            start,
            0.04,
            grad_double_well,
            35,
        )
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=3,
            marker="o",
            markersize=5,
            markevery=max(1, len(path) // 8),
            label=label,
        )
        ax.scatter(
            [path[0, 0]],
            [path[0, 1]],
            marker="x",
            s=180,
            linewidths=3,
            color=color,
            zorder=4,
        )
    ax.scatter(
        [minima[0], minima[1]],
        [0, 0],
        marker="*",
        s=300,
        color="#111827",
        zorder=5,
    )
    ax.annotate(
        "global minimum",
        xy=(minima[0], 0),
        xytext=(-1.68, -0.95),
        fontsize=LABEL_SIZE,
        arrowprops={"arrowstyle": "->", "lw": 1.5},
    )
    ax.annotate(
        "higher local minimum",
        xy=(minima[1], 0),
        xytext=(0.35, 0.9),
        fontsize=LABEL_SIZE,
        arrowprops={"arrowstyle": "->", "lw": 1.5},
    )
    ax.set(
        title=r"Nonconvex landscape: $f(x,y)=(x^2-1)^2+0.2x+y^2$",
        xlabel="x",
        ylabel="y",
        xlim=x_limits,
        ylim=y_limits,
    )
    _style_axes(ax, equal=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    return fig


def make_saddle():
    """Show a stationary saddle with a sparse gradient field."""

    x_limits = (-2.2, 2.2)
    y_limits = (-2.0, 2.0)
    coordinates_x = np.linspace(*x_limits, 11)
    coordinates_y = np.linspace(*y_limits, 9)
    x_grid, y_grid = np.meshgrid(coordinates_x, coordinates_y)
    grad_x = 2 * x_grid
    grad_y = -2 * y_grid
    magnitude = np.hypot(grad_x, grad_y)
    safe_magnitude = np.where(magnitude == 0, 1.0, magnitude)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    x_values, y_values, values = _objective_grid(
        "saddle",
        x_limits,
        y_limits,
    )
    ax.contour(
        x_values,
        y_values,
        values,
        levels=(-6, -3, -1, 1, 3, 6),
        colors="#64748b",
        linewidths=1.1,
        alpha=0.62,
    )
    ax.quiver(
        x_grid,
        y_grid,
        grad_x / safe_magnitude,
        grad_y / safe_magnitude,
        color="#374151",
        angles="xy",
        scale_units="xy",
        scale=2.2,
        width=0.0045,
        headwidth=4.5,
        headlength=5.5,
    )
    ax.scatter(
        [0],
        [0],
        marker="X",
        s=260,
        color="#dc2626",
        edgecolor="white",
        linewidth=1.2,
        zorder=5,
    )
    ax.text(
        -1.95,
        -1.72,
        r"$\nabla f=0$ is not enough",
        fontsize=LABEL_SIZE,
        color="#991b1b",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#fca5a5"},
    )
    ax.set(
        title=r"Saddle point: $f(x,y)=x^2-y^2$",
        xlabel="x",
        ylabel="y",
        xlim=x_limits,
        ylim=y_limits,
    )
    _style_axes(ax, equal=True)
    return fig


def _box(ax, x, y, width, height, title, subtitle, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=color,
        edgecolor="#334155",
        linewidth=1.7,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        x + width / 2,
        y + height * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=11.5,
        color="#334155",
    )


def _arrow(ax, start, end, *, dashed: bool = False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            color="#475569",
            linestyle="--" if dashed else "-",
        )
    )


def make_optimizer_family_map():
    """Build a compact story map with explicit problem-to-method links."""

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title(
        "Optimizer story: each variant answers a concrete weakness",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    colors = {
        "base": "#dbeafe",
        "memory": "#dcfce7",
        "scale": "#fef3c7",
        "combine": "#f3e8ff",
        "regularize": "#fee2e2",
        "advanced": "#e2e8f0",
    }

    _box(ax, 0.25, 3.0, 1.65, 1.0, "Gradient", "local slope", colors["base"])
    _box(ax, 2.65, 4.55, 2.05, 1.0, "Full-batch GD", "exact, expensive", colors["base"])
    _box(ax, 2.65, 2.45, 2.05, 1.0, "Mini-batch SGD", "scalable, noisy", colors["base"])
    _box(
        ax, 5.55, 4.55, 2.25, 1.0, "Momentum / NAG", "smooth zigzags", colors["memory"]
    )
    _box(
        ax,
        5.55,
        2.45,
        2.25,
        1.0,
        "AdaGrad / RMSProp",
        "adaptive scale",
        colors["scale"],
    )
    _box(ax, 8.65, 3.50, 1.65, 1.0, "Adam", "memory + scale", colors["combine"])
    _box(ax, 11.0, 3.50, 1.55, 1.0, "AdamW", "decoupled decay", colors["regularize"])
    _box(
        ax,
        8.65,
        1.15,
        2.25,
        1.0,
        "Lion / SAM / L-BFGS",
        "appendix ideas",
        colors["advanced"],
    )

    _arrow(ax, (1.9, 3.5), (2.62, 5.05))
    _arrow(ax, (1.9, 3.5), (2.62, 2.95))
    _arrow(ax, (4.72, 2.95), (5.52, 5.05))
    _arrow(ax, (4.72, 2.95), (5.52, 2.95))
    _arrow(ax, (7.83, 5.05), (8.62, 4.1))
    _arrow(ax, (7.83, 2.95), (8.62, 3.9))
    _arrow(ax, (10.32, 4.0), (10.97, 4.0))
    _arrow(ax, (9.47, 3.45), (9.47, 2.2), dashed=True)
    return fig


def make_cnn_architecture():
    """Build a general CNN feature-learning and classification diagram."""

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title(
        "NumPy CNN for MNIST: feature learning → classification",
        fontsize=TITLE_SIZE,
        pad=12,
    )

    def feature_maps(x, y, width, height, color, title, subtitle, depth=3):
        for offset in np.linspace(0.18, 0.0, depth):
            ax.add_patch(
                Rectangle(
                    (x + offset, y + offset),
                    width,
                    height,
                    facecolor="#ffffff",
                    edgecolor=color,
                    linewidth=1.7,
                    alpha=0.82,
                    zorder=2,
                )
            )
        ax.text(
            x + width / 2,
            y + height + 0.38,
            title,
            ha="center",
            va="center",
            fontsize=12.5,
            fontweight="bold",
        )
        ax.text(
            x + width / 2,
            y - 0.34,
            subtitle,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#334155",
        )

    def dense_nodes(x, y_values, color, title, subtitle):
        for y_value in y_values:
            ax.add_patch(
                Circle(
                    (x, y_value),
                    radius=0.13,
                    facecolor=color,
                    edgecolor="#334155",
                    linewidth=1.2,
                    zorder=3,
                )
            )
        ax.text(
            x, 5.98, title, ha="center", va="center", fontsize=11.5, fontweight="bold"
        )
        ax.text(
            x, 2.55, subtitle, ha="center", va="center", fontsize=10.5, color="#334155"
        )

    # A deliberately simple digit keeps the input recognizable without making
    # the architecture figure depend on a downloaded MNIST sample.
    digit = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
        ]
    )
    ax.imshow(
        digit,
        cmap="Blues",
        interpolation="nearest",
        extent=(0.45, 1.65, 3.0, 5.0),
        vmin=0,
        vmax=1,
        zorder=2,
    )
    ax.add_patch(
        Rectangle((0.45, 3.0), 1.2, 2.0, fill=False, edgecolor="#334155", linewidth=1.7)
    )
    ax.text(
        1.05,
        5.42,
        "Input image",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.text(
        1.05, 2.55, "1×28×28", ha="center", va="center", fontsize=10.5, color="#334155"
    )

    feature_maps(
        2.45, 3.35, 1.25, 1.55, "#0f766e", "Convolution + ReLU", "16 feature maps"
    )
    feature_maps(4.25, 3.65, 0.95, 1.15, "#d97706", "Pooling", "16×14×14", depth=2)
    feature_maps(
        5.85, 3.35, 1.25, 1.55, "#0f766e", "Convolution + ReLU", "32 feature maps"
    )
    feature_maps(7.65, 3.65, 0.95, 1.15, "#d97706", "Pooling", "32×7×7", depth=2)

    _box(ax, 9.15, 3.55, 1.05, 1.35, "Flatten", "1568", "#e0e7ff")
    dense_nodes(11.15, (4.85, 4.3, 3.75, 3.2), "#c4b5fd", "Dense", "128 units")
    dense_nodes(12.95, (4.55, 3.95, 3.35), "#f9a8d4", "Logits", "10 classes")

    arrow_y = 4.15
    arrow_pairs = (
        (1.7, 2.35),
        (3.8, 4.15),
        (5.3, 5.75),
        (7.3, 7.55),
        (8.8, 9.05),
        (10.3, 11.0),
        (11.3, 12.8),
    )
    for start_x, end_x in arrow_pairs:
        _arrow(ax, (start_x, arrow_y), (end_x, arrow_y))

    for source_y in (4.85, 4.3, 3.75, 3.2):
        for target_y in (4.55, 3.95, 3.35):
            ax.plot(
                (11.28, 12.82),
                (source_y, target_y),
                color="#94a3b8",
                linewidth=0.8,
                alpha=0.55,
                zorder=1,
            )

    # Dropout masks activations for one training pass; it does not remove the
    # parameter or permanently change the network architecture.
    dropout_node = (11.15, 3.2)
    ax.add_patch(
        Circle(
            dropout_node,
            radius=0.13,
            facecolor="#e5e7eb",
            edgecolor="#6b7280",
            linewidth=1.2,
            zorder=4,
        )
    )
    ax.plot((11.05, 11.25), (3.1, 3.3), color="#6b7280", linewidth=1.8, zorder=5)
    ax.plot((11.05, 11.25), (3.3, 3.1), color="#6b7280", linewidth=1.8, zorder=5)
    ax.text(
        11.15,
        2.12,
        "dropout: one activation masked\n(training only)",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#6b7280",
    )

    ax.text(
        5.0,
        1.65,
        "FEATURE LEARNING",
        ha="center",
        va="center",
        fontsize=13,
        color="#0f766e",
        fontweight="bold",
    )
    ax.plot((1.8, 8.35), (1.95, 1.95), color="#0f766e", linewidth=2.2)
    ax.plot((1.8, 1.8), (1.95, 2.15), color="#0f766e", linewidth=2.2)
    ax.plot((8.35, 8.35), (1.95, 2.15), color="#0f766e", linewidth=2.2)
    ax.text(
        11.95,
        1.65,
        "CLASSIFICATION",
        ha="center",
        va="center",
        fontsize=13,
        color="#7c3aed",
        fontweight="bold",
    )
    ax.plot((9.0, 13.55), (1.95, 1.95), color="#7c3aed", linewidth=2.2)
    ax.plot((9.0, 9.0), (1.95, 2.15), color="#7c3aed", linewidth=2.2)
    ax.plot((13.55, 13.55), (1.95, 2.15), color="#7c3aed", linewidth=2.2)

    ax.text(
        7.3,
        0.88,
        "forward pass → loss → backpropagation → selected optimizer updates W and b",
        ha="center",
        va="center",
        fontsize=14.5,
        color="#334155",
    )
    return fig


def make_training_loss_template():
    """Build a clearly labelled, non-result template for future metrics."""

    steps = np.arange(0, 151)
    curves = {
        "SGD": (2.55 * np.exp(-steps / 82) + 0.25, "#2563eb"),
        "Momentum": (2.4 * np.exp(-steps / 58) + 0.17, "#ea580c"),
        "RMSProp": (2.3 * np.exp(-steps / 42) + 0.12, "#16a34a"),
        "Adam": (2.25 * np.exp(-steps / 31) + 0.10, "#dc2626"),
    }
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    for label, (curve, color) in curves.items():
        noisy = curve + rng.normal(0, 0.018, size=steps.shape)
        ax.plot(steps, noisy, linewidth=3, color=color, label=label)
    ax.set(
        title="Illustrative layout only — replace with measured MNIST metrics",
        xlabel="optimizer step",
        ylabel="training loss",
        xlim=(0, 150),
        ylim=(0, 2.8),
    )
    _style_axes(ax)
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE, frameon=False)
    ax.text(
        3,
        0.18,
        "Not experimental evidence",
        fontsize=LABEL_SIZE,
        color="#991b1b",
        bbox={"boxstyle": "round,pad=0.25", "fc": "#fef2f2", "ec": "#fca5a5"},
    )
    return fig


def generate_all_figures(png_dir: Path, svg_dir: Path) -> None:
    """Regenerate the ten prepared seminar figures."""

    builders = (
        ("01_gradient_field_sphere", make_gradient_field),
        ("02_surface_sphere_3d", make_surface),
        ("03_gd_steps_sphere", make_gd_steps),
        ("04_learning_rate_ill_conditioned", make_learning_rate),
        ("05_optimizer_trajectories_ill_conditioned", make_optimizer_comparison),
        ("06_nonconvex_local_minima", make_nonconvex),
        ("07_saddle_point", make_saddle),
        ("08_optimizer_family_map", make_optimizer_family_map),
        ("09_numpy_cnn_architecture", make_cnn_architecture),
        ("10_training_loss_template", make_training_loss_template),
    )
    for name, builder in builders:
        figure = builder()
        _save(
            figure,
            png_dir / f"{name}.png",
            svg_dir / f"{name}.svg",
        )
