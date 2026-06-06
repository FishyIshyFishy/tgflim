"""Parameter containers for time-gated FLIM simulation and analysis.

All times are in nanoseconds unless a field name says otherwise. There is no
ps<->ns juggling anywhere in the package; convert at your script boundary if a
value is more natural in picoseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import grid


class ExecMode(Enum):
    """How per-pixel work is dispatched.

    serial      -> plain Python loop over pixels.
    vectorized  -> single batched array operation (closed-form RLD fitting).
    pool        -> process pool over pixels (curve-fit based methods, generation).
    """

    serial = "serial"
    vectorized = "vectorized"
    pool = "pool"


@dataclass
class Acquisition:
    """Time-gated SPAD acquisition settings."""

    freq: float                          # laser repetition rate, MHz
    width: float                         # gate width, ns
    step: float                          # gate step increment, ns
    offset: float = 0.0                  # opening time of the first gate, ns
    bits: int = 8                        # bit depth; saturation level is 2**bits - 1
    integ: float = 1000.0                # per-gate integration / exposure, microseconds
    numsteps: int = 0                    # number of gates; 0 fills the interpulse period
    kernel_size: int = 0                 # spatial-binning half-width used by bit-depth correction
    exec_mode: ExecMode = ExecMode.serial
    workers: int = 8                     # process count when exec_mode is pool

    def resolved_numsteps(self) -> int:
        return grid.numsteps_for(self)

    def time_axis(self) -> np.ndarray:
        return grid.time_axis(self)


@dataclass
class Sample:
    """Fluorophore ground truth: a (multi-)exponential decay and a brightness scale."""

    lifetimes: list[float]                              # decay constants, ns (one per component)
    weights: list[float] = field(default_factory=lambda: [1.0])  # fractional amplitudes, sum to 1
    zeta: float = 1.0                                   # detection efficiency / brightness, 0 < zeta <= 1


@dataclass
class Detector:
    """Detection-side imperfections shared across a field of view."""

    irf_mean: float = 0.0                # mean of the pulse/gate origin timing jitter, ns
    irf_width: float = 0.0               # stdev of the timing jitter, ns (0 -> ideal timing)
    dark_cps: float = 0.0                # dark count rate, counts per second
