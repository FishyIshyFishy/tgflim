"""Derived acquisition quantities, computed in one place so the forward model,
fitter, and corrector never disagree on gate counts or the time axis.

All inputs are read off an ``Acquisition``; everything is nanoseconds. Note that
``freq`` is in MHz and ``integ`` in microseconds, so ``freq * integ`` is already a
dimensionless cycle count.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .params import Acquisition


def numsteps_for(acq: Acquisition) -> int:
    if acq.numsteps:
        return acq.numsteps
    return int(1e3 / (acq.freq * acq.step))


def time_axis(acq: Acquisition) -> np.ndarray:
    return np.arange(numsteps_for(acq)) * acq.step + acq.offset


def numgates(acq: Acquisition) -> int:
    """Laser cycles integrated at each gate position."""
    return int(acq.freq * acq.integ)


def max_levels(acq: Acquisition) -> int:
    """Largest count a single (binned) pixel can report, 2**bits - 1."""
    return 2 ** acq.bits - 1


def bin_gates(acq: Acquisition) -> int:
    """Laser cycles accumulated into each of the ``max_levels`` sub-frames."""
    return int(numgates(acq) / max_levels(acq))
