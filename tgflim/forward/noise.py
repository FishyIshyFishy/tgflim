"""Stochastic detection layer: bit-depth saturation, dark counts, photon statistics.

A bit-depth-limited SPAD reports, per pixel, the number of sub-frames (out of
``2**bits - 1``) that registered at least one event. Each sub-frame accumulates
``bin_gates`` laser cycles. Both signal and dark counts can trigger a sub-frame, so
they combine at the per-sub-frame probability level and saturate together -- this is
why dark counts are folded in here rather than added afterwards.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import grid

if TYPE_CHECKING:
    from ..params import Acquisition, Detector


def signal_subframe_prob(gate_prob: np.ndarray, acq: Acquisition) -> np.ndarray:
    """Probability a sub-frame registers signal: ``1 - (1 - p)**bin_gates``."""
    return 1.0 - (1.0 - gate_prob) ** grid.bin_gates(acq)


def dark_subframe_prob(acq: Acquisition, detector: Detector) -> float:
    """Probability a sub-frame registers a dark count over its open time."""
    open_time_s = grid.bin_gates(acq) * acq.width * 1e-9
    return 1.0 - np.exp(-detector.dark_cps * open_time_s)


def combine_independent(p_signal: np.ndarray, p_dark: float) -> np.ndarray:
    return 1.0 - (1.0 - p_signal) * (1.0 - p_dark)


def sample_gated_counts(subframe_prob: np.ndarray, acq: Acquisition, rng: np.random.Generator) -> np.ndarray:
    """Draw the bit-depth-limited count per gate from the binomial detector model."""
    return rng.binomial(grid.max_levels(acq), subframe_prob)


def sample_photon_counts(gate_prob: np.ndarray, acq: Acquisition, rng: np.random.Generator) -> np.ndarray:
    """Draw counts in the unsaturated (photon-counting) limit over all laser cycles."""
    return rng.binomial(grid.numgates(acq), gate_prob)
