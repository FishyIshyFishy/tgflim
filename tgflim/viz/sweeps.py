"""Parameter-sweep heatmaps: a fitted quantity over a 2D grid of swept parameters."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .cmaps import centered_norm, diverging_cmap, sequential_cmap

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_sweep(ax: Axes, grid: np.ndarray, *, xticks: np.ndarray, yticks: np.ndarray, xlabel: str, ylabel: str, cbar_label: str, center: float | None = None, half_range: float = 2.5) -> None:
    if center is None:
        cmap, norm = sequential_cmap(), None
    else:
        cmap, norm = diverging_cmap(), centered_norm(center, half_range)
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm, extent=(xticks[0], xticks[-1], yticks[0], yticks[-1]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.figure.colorbar(image, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
