"""Bit-depth correction: map saturated gate counts back toward detection probability.

A bit-depth-limited measurement compresses counts as they approach ``max_counts``.
These inverses undo that compression before fitting. ``antol`` uses the logarithmic
inverse; ``binomial`` inverts the per-sub-frame binomial saturation exactly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import grid

if TYPE_CHECKING:
    from ..params import Acquisition


def max_counts(acq: Acquisition) -> float:
    return ((1 + 2 * acq.kernel_size) ** 2) * grid.max_levels(acq)


def correct_bit_depth(counts: np.ndarray, acq: Acquisition, *, method: str) -> np.ndarray:
    ceiling = max_counts(acq)
    probs = counts / ceiling
    if method == "antol":
        return -ceiling * np.log(1 - probs + 1e-9)
    if method == "binomial":
        probs = 1 - (1 - probs) ** (1 / grid.bin_gates(acq))
        return ceiling * probs
    raise ValueError(f"unknown correction method {method!r}; choose 'antol' or 'binomial'")
