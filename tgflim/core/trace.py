"""The primitive unit of simulation and analysis: one gated decay curve."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..params import Sample


class PixelTrace:
    """A single gated decay: gate-opening times and the counts measured in each gate.

    Carries optional ``truth`` (the ``Sample`` it was generated from) so figures can
    compare a fit against ground truth. Knows nothing about how it was made or fit.
    """

    def __init__(self, times: np.ndarray, counts: np.ndarray, truth: Sample | None = None) -> None:
        self.times = np.asarray(times, dtype=float)
        self.counts = np.asarray(counts)
        self.truth = truth

    @property
    def total(self) -> float:
        return float(np.sum(self.counts))

    @classmethod
    def from_arrays(cls, times: np.ndarray, counts: np.ndarray) -> PixelTrace:
        return cls(times=times, counts=counts)

    def __len__(self) -> int:
        return len(self.counts)
