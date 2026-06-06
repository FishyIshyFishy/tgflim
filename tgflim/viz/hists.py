"""Lifetime histograms, including a broken-axis variant for well-separated components."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def clean(values: np.ndarray) -> np.ndarray:
    values = np.ravel(values)
    return values[np.isfinite(values) & (values > 0)]


def plot_lifetime_hist(ax: Axes, taus: dict[str, np.ndarray], bins: int = 30) -> None:
    for name, values in taus.items():
        ax.hist(clean(values), bins=bins, alpha=0.6, label=name)
    ax.set_xlabel("lifetime (ns)")
    ax.set_ylabel("count")
    ax.legend()


def plot_lifetime_hist_broken(taus: dict[str, np.ndarray], *, splice: tuple[float, float], bins: int = 30) -> tuple[Figure, tuple[Axes, Axes]]:
    """Two panels sharing a y-axis with a break between ``splice[0]`` and ``splice[1]``."""
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
    for name, values in taus.items():
        data = clean(values)
        left.hist(data, bins=bins, alpha=0.6, label=name)
        right.hist(data, bins=bins, alpha=0.6, label=name)
    left.set_xlim(right=splice[0])
    right.set_xlim(left=splice[1])
    left.spines["right"].set_visible(False)
    right.spines["left"].set_visible(False)
    right.tick_params(left=False)
    left.set_xlabel("lifetime (ns)")
    right.set_xlabel("lifetime (ns)")
    left.set_ylabel("count")
    left.legend()
    return fig, (left, right)
