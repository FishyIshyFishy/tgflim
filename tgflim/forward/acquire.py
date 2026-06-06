"""Acquisition modalities. The forward simulation is the only modality-specific part
of the package; analysis and plotting operate on a plain ``PixelTrace``. A future
TCSPC model slots in here as another ``AcquisitionModel`` without touching ``fit/``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from . import model, noise

if TYPE_CHECKING:
    from ..params import Acquisition, Detector, Sample


class AcquisitionModel(ABC):
    @abstractmethod
    def simulate(self, sample: Sample, acq: Acquisition, detector: Detector, rng: np.random.Generator) -> np.ndarray:
        """Return the per-gate counts measured for one sample."""


class TimeGated(AcquisitionModel):
    """Bit-depth-limited time-gated detection (the SPAD512 modality)."""

    def simulate(self, sample: Sample, acq: Acquisition, detector: Detector, rng: np.random.Generator) -> np.ndarray:
        times = acq.time_axis()
        prob = model.gated_prob(times, sample=sample, detector=detector, width=acq.width)
        p_signal = noise.signal_subframe_prob(prob, acq)
        p_dark = noise.dark_subframe_prob(acq, detector)
        p_subframe = noise.combine_independent(p_signal, p_dark)
        return noise.sample_gated_counts(p_subframe, acq, rng)
