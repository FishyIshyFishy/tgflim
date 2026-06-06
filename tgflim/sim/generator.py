"""Drives an acquisition model over a single sample (one trace) or an image of
ground-truth samples (a cube), honouring the acquisition's ``exec_mode``.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from ..core import Image, PixelTrace
from ..params import Acquisition, ExecMode

if TYPE_CHECKING:
    from ..forward.acquire import AcquisitionModel
    from ..params import Detector, Sample


def simulate_seeded(args: tuple) -> tuple[int, int, np.ndarray]:
    """Top-level worker for process-pool generation (must be picklable)."""
    model, sample, acq, detector, i, j, seed = args
    rng = np.random.default_rng(seed)
    return i, j, model.simulate(sample=sample, acq=acq, detector=detector, rng=rng)


class Generator:
    def __init__(self, acq: Acquisition, model: AcquisitionModel, seed: int | None = None) -> None:
        self.acq = acq
        self.model = model
        self.rng = np.random.default_rng(seed)

    def trace(self, sample: Sample, detector: Detector) -> PixelTrace:
        counts = self.model.simulate(sample=sample, acq=self.acq, detector=detector, rng=self.rng)
        return PixelTrace(times=self.acq.time_axis(), counts=counts, truth=sample)

    def image(self, image: Image) -> np.ndarray:
        if self.acq.exec_mode is ExecMode.pool:
            return self.image_pool(image)
        return self.image_serial(image)

    def image_serial(self, image: Image) -> np.ndarray:
        numsteps = self.acq.resolved_numsteps()
        x, y = image.shape
        cube = np.zeros((numsteps, x, y), dtype=int)
        for i, j, sample in image.iter_pixels():
            cube[:, i, j] = self.model.simulate(sample=sample, acq=self.acq, detector=image.detector, rng=self.rng)
        return cube

    def image_pool(self, image: Image) -> np.ndarray:
        numsteps = self.acq.resolved_numsteps()
        x, y = image.shape
        cube = np.zeros((numsteps, x, y), dtype=int)
        seeds = self.rng.integers(0, 2**63 - 1, size=(x, y))
        tasks = [(self.model, sample, self.acq, image.detector, i, j, int(seeds[i, j])) for i, j, sample in image.iter_pixels()]
        with ProcessPoolExecutor(max_workers=self.acq.workers) as pool:
            for i, j, counts in pool.map(simulate_seeded, tasks):
                cube[:, i, j] = counts
        return cube
