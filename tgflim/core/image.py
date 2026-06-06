"""A spatial field of ground-truth samples for FLIM image / super-resolution studies.

An ``Image`` holds 2D maps of the decay parameters so different regions can carry
different lifetimes and weights, plus one shared ``Acquisition`` and ``Detector``
(timing jitter and dark counts are properties of the instrument, not the pixel).
It is physics-free: the ``Generator`` turns it into a ``(numsteps, x, y)`` cube.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np

from ..params import Sample

if TYPE_CHECKING:
    from ..params import Acquisition, Detector


class Image:
    def __init__(
        self,
        acq: Acquisition,
        detector: Detector,
        lifetime_maps: np.ndarray,                 # (ncomp, x, y), ns
        weight_maps: np.ndarray,                   # (ncomp, x, y), fractional amplitudes
        zeta_map: np.ndarray | None = None,        # (x, y), per-pixel efficiency
        amplitude_map: np.ndarray | None = None,   # (x, y), extra brightness multiplier
    ) -> None:
        self.acq = acq
        self.detector = detector
        self.lifetime_maps = np.asarray(lifetime_maps, dtype=float)
        self.weight_maps = np.asarray(weight_maps, dtype=float)
        self.zeta_map = zeta_map
        self.amplitude_map = amplitude_map

    @property
    def ncomp(self) -> int:
        return self.lifetime_maps.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return self.lifetime_maps.shape[1], self.lifetime_maps.shape[2]

    def zeta_at(self, i: int, j: int) -> float:
        zeta = 1.0 if self.zeta_map is None else float(self.zeta_map[i, j])
        if self.amplitude_map is not None:
            zeta *= float(self.amplitude_map[i, j])
        return zeta

    def sample_at(self, i: int, j: int) -> Sample:
        return Sample(
            lifetimes=list(self.lifetime_maps[:, i, j]),
            weights=list(self.weight_maps[:, i, j]),
            zeta=self.zeta_at(i, j),
        )

    def iter_pixels(self) -> Iterator[tuple[int, int, Sample]]:
        x, y = self.shape
        for i in range(x):
            for j in range(y):
                yield i, j, self.sample_at(i, j)

    def truth_maps(self) -> dict[str, np.ndarray]:
        maps: dict[str, np.ndarray] = {}
        for k in range(self.ncomp):
            maps[f"tau{k + 1}"] = self.lifetime_maps[k]
            maps[f"weight{k + 1}"] = self.weight_maps[k]
        return maps
