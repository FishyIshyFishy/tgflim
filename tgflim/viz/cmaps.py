"""Colormaps and norms for FLIM figures.

A diverging norm centered on the ground-truth lifetime makes over- and
under-estimation read at a glance; a sequential norm suits precision/uncertainty maps.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


def centered_norm(center: float, half_range: float) -> TwoSlopeNorm:
    return TwoSlopeNorm(vmin=center - half_range, vcenter=center, vmax=center + half_range)


def linear_norm(vmin: float, vmax: float) -> Normalize:
    return Normalize(vmin=vmin, vmax=vmax)


def diverging_cmap():
    return plt.get_cmap("seismic")


def sequential_cmap():
    return plt.get_cmap("plasma")


def intensity_cmap():
    return plt.get_cmap("gray")
