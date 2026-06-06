"""Spatial parameter maps (lifetime, amplitude, intensity) from fit results."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from .cmaps import diverging_cmap, sequential_cmap

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_param_map(ax: Axes, data: np.ndarray, *, title: str, cbar_label: str, cmap=None, norm=None) -> None:
    image = ax.imshow(data, cmap=cmap or sequential_cmap(), norm=norm)
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)


def plot_result_maps(results: dict[str, np.ndarray], *, keys: list[str] | None = None) -> tuple[Figure, np.ndarray]:
    """Grid of maps, one per requested result key (defaults to every lifetime map)."""
    keys = keys or [k for k in results if k.startswith("tau")]
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        cmap = diverging_cmap() if key.startswith("tau") else sequential_cmap()
        plot_param_map(ax, results[key], title=key, cbar_label="ns" if key.startswith("tau") else "", cmap=cmap)
    return fig, axes[0]
