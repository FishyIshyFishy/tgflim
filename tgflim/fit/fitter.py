"""Apply a registered fit method to a single trace or a whole image.

Dispatches on ``Acquisition.exec_mode``: ``vectorized`` solves all pixels in one
closed-form call (RLD only), ``pool`` farms curve-fit work to a process pool, and
``serial`` loops. Component sorting (ascending lifetime) lives here once, used by both
the scalar and the map code paths.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

from ..core import PixelTrace
from ..params import Acquisition, Detector, ExecMode
from . import registry
from .correct import correct_bit_depth
from .rld import bi_rld_kernel

if TYPE_CHECKING:
    from ..params import Sample

vectorizable = {"mono_rld", "mono_rld_50ovp", "bi_rld"}


def sort_components(result: dict[str, float]) -> dict[str, float]:
    if "tau1" in result and "tau2" in result and result["tau1"] > result["tau2"]:
        result = dict(result)
        result["tau1"], result["tau2"] = result["tau2"], result["tau1"]
        result["A1"], result["A2"] = result["A2"], result["A1"]
    return result


def sort_component_maps(maps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "tau1" not in maps or "tau2" not in maps:
        return maps
    swap = maps["tau1"] > maps["tau2"]
    out = {k: v.copy() for k, v in maps.items()}
    for lo, hi in (("tau1", "tau2"), ("A1", "A2")):
        out[lo][swap], out[hi][swap] = maps[hi][swap], maps[lo][swap]
    return out


def build_maps(results: dict[tuple[int, int], dict[str, float]], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    if not results:
        return {}
    keys = next(iter(results.values())).keys()
    maps = {k: np.zeros(shape, dtype=float) for k in keys}
    for (i, j), res in results.items():
        for k, v in res.items():
            maps[k][i, j] = v
    return maps


def fit_pixel_worker(args: tuple) -> tuple[int, int, dict[str, float] | None]:
    """Top-level worker for process-pool fitting (must be picklable)."""
    method, times, counts, acq, detector, i, j = args
    res = registry.get(method)(times, counts, acq, detector, None)
    return i, j, sort_components(res) if res else None


class Fitter:
    def __init__(self, acq: Acquisition, method: str, *, thresh: float = 0.0, correct: str | None = None, detector: Detector | None = None) -> None:
        self.acq = acq
        self.method = method
        self.thresh = thresh
        self.correct = correct
        self.detector = detector
        self.func = registry.get(method)

    def prep(self, counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=float)
        if self.correct:
            return correct_bit_depth(counts, self.acq, method=self.correct)
        return counts

    def fit_trace(self, trace: PixelTrace) -> dict[str, float] | None:
        counts = self.prep(trace.counts)
        if np.sum(counts) < self.thresh:
            return None
        res = self.func(trace.times, counts, self.acq, self.detector, trace.truth)
        return sort_components(res) if res else None

    def fit_image(self, cube: np.ndarray) -> dict[str, np.ndarray]:
        if self.acq.exec_mode is ExecMode.vectorized and self.method in vectorizable:
            return self.fit_image_vectorized(cube)
        if self.acq.exec_mode is ExecMode.pool:
            return self.fit_image_pool(cube)
        return self.fit_image_serial(cube)

    def fit_image_serial(self, cube: np.ndarray) -> dict[str, np.ndarray]:
        times = self.acq.time_axis()
        x, y = cube.shape[1], cube.shape[2]
        results = {}
        for i in range(x):
            for j in range(y):
                res = self.fit_trace(PixelTrace(times, cube[:, i, j]))
                if res:
                    results[(i, j)] = res
        return build_maps(results, (x, y))

    def fit_image_pool(self, cube: np.ndarray) -> dict[str, np.ndarray]:
        times = self.acq.time_axis()
        x, y = cube.shape[1], cube.shape[2]
        tasks = []
        for i in range(x):
            for j in range(y):
                counts = self.prep(cube[:, i, j])
                if np.sum(counts) >= self.thresh:
                    tasks.append((self.method, times, counts, self.acq, self.detector, i, j))
        results = {}
        with ProcessPoolExecutor(max_workers=self.acq.workers) as pool:
            for i, j, res in pool.map(fit_pixel_worker, tasks):
                if res:
                    results[(i, j)] = res
        return build_maps(results, (x, y))

    def fit_image_vectorized(self, cube: np.ndarray) -> dict[str, np.ndarray]:
        data = self.prep(cube)
        mask = data.sum(axis=0) >= self.thresh
        if self.method == "bi_rld":
            a1, tau1, a2, tau2 = bi_rld_kernel(data[:4], width=self.acq.width, step=self.acq.step)
            maps = sort_component_maps({"A1": a1, "tau1": tau1, "A2": a2, "tau2": tau2})
        else:
            maps = self.mono_rld_maps(data)
        valid = mask & np.all([np.isfinite(v) for v in maps.values()], axis=0)
        return {k: np.where(valid, v, 0.0) for k, v in maps.items()}

    def mono_rld_maps(self, data: np.ndarray) -> dict[str, np.ndarray]:
        d0, d1 = data[0], data[1]
        if self.method == "mono_rld_50ovp":
            tau = -self.acq.step / np.log((d1**2) / (d0**2))
            amp = 2 * (d0**3) * np.log(d1 / d0) / (self.acq.step * ((d1**2) - (d0**2)))
        else:
            ratio = np.log(d0 / d1)
            tau = self.acq.step / ratio
            amp = (d0**2) * ratio / (self.acq.step * (d0 - d1))
        return {"A": amp, "tau": tau}
