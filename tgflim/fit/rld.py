"""Rapid Lifetime Determination: closed-form estimators from a few gates.

These are the only place the bi-exponential RLD algebra lives (the SPAD512 code
duplicated it across five files). ``bi_rld_kernel`` is written to broadcast over a
trailing pixel axis so a whole image is solved in one call by the Fitter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .registry import register

if TYPE_CHECKING:
    from ..params import Acquisition, Detector, Sample


@register("mono_rld")
def mono_rld(times, counts, acq: Acquisition, detector: Detector | None = None, sample: Sample | None = None) -> dict[str, float]:
    d0, d1 = counts[0], counts[1]
    ratio = np.log(d0 / d1)
    tau = acq.step / ratio
    amp = (d0**2) * ratio / (acq.step * (d0 - d1))
    return {"A": float(amp), "tau": float(tau)}


@register("mono_rld_50ovp")
def mono_rld_50ovp(times, counts, acq: Acquisition, detector: Detector | None = None, sample: Sample | None = None) -> dict[str, float]:
    d0, d1 = counts[0], counts[1]
    tau = -acq.step / np.log((d1**2) / (d0**2))
    amp = 2 * (d0**3) * np.log(d1 / d0) / (acq.step * ((d1**2) - (d0**2)))
    return {"A": float(amp), "tau": float(tau)}


def bi_rld_kernel(gates: np.ndarray, width: float, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the 4-gate bi-exponential RLD. ``gates`` is ``(4,)`` or ``(4, npix)``."""
    d0, d1, d2, d3 = gates
    ratio = width / step

    r = d1 * d1 - d2 * d0
    p = d3 * d0 - d2 * d1
    q = d2 * d2 - d3 * d1
    disc = p**2 - 4 * r * q
    root = np.sqrt(disc)
    y = (-p + root) / (2 * r)
    x = (-p - root) / (2 * r)

    s = step * (x**2 * d0 - 2 * x * d1 + d2)
    t = (1 - ((x * d1 - d2) / (x * d0 - d1))) ** ratio

    tau1 = -step / np.log(y)
    tau2 = -step / np.log(x)
    a1 = (-((x * d0 - d1) ** 2)) * np.log(y) / (s * t)
    a2 = (-r * np.log(x)) / (s * (x**ratio - 1))
    return a1, tau1, a2, tau2


@register("bi_rld")
def bi_rld(times, counts, acq: Acquisition, detector: Detector | None = None, sample: Sample | None = None) -> dict[str, float]:
    a1, tau1, a2, tau2 = bi_rld_kernel(np.asarray(counts[:4], dtype=float), width=acq.width, step=acq.step)
    return {"A1": float(a1), "tau1": float(tau1), "A2": float(a2), "tau2": float(tau2)}
