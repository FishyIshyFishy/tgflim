"""Nonlinear least-squares exponential fits.

The IRF-convolved variants reuse ``forward.model.emg_kernel`` so the simulation and
the fit speak the exact same exponentially-modified-Gaussian, parameterised by the
detector's timing jitter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

from ..forward.model import emg_kernel, gate_integral
from .registry import register

if TYPE_CHECKING:
    from ..params import Acquisition, Detector, Sample


def mono_model(t: np.ndarray, amp: float, tau: float) -> np.ndarray:
    return amp * np.exp(-t / tau)


def bi_model(t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float) -> np.ndarray:
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)


def gated_emg(t: np.ndarray, tau: float, irf_mean: float, irf_width: float, width: float) -> np.ndarray:
    """EMG intensity integrated over the gate window, matching the forward model."""
    return gate_integral(t, width=width, rate_fn=lambda u: emg_kernel(u, tau=tau, irf_mean=irf_mean, irf_width=irf_width), substeps=8)


def mono_conv_model(t: np.ndarray, amp: float, tau: float, irf_mean: float, irf_width: float, width: float) -> np.ndarray:
    return amp * gated_emg(t, tau=tau, irf_mean=irf_mean, irf_width=irf_width, width=width)


def bi_conv_model(t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float, irf_mean: float, irf_width: float, width: float) -> np.ndarray:
    return a1 * gated_emg(t, tau=tau1, irf_mean=irf_mean, irf_width=irf_width, width=width) + a2 * gated_emg(t, tau=tau2, irf_mean=irf_mean, irf_width=irf_width, width=width)


def rising_edge(counts: np.ndarray, keep: int) -> int:
    """Index of the decay peak, clamped so at least ``keep`` points remain."""
    return min(int(np.argmax(counts)), len(counts) - keep)


@register("mono")
def mono(times, counts, acq: Acquisition, detector: Detector | None = None, sample: Sample | None = None) -> dict[str, float] | None:
    loc = rising_edge(counts, keep=2)
    guess = (float(np.max(counts)), 2.0)
    try:
        popt, _ = curve_fit(mono_model, times[loc:], counts[loc:], p0=guess, maxfev=5000)
    except (RuntimeError, ValueError):
        return None
    return {"A": float(popt[0]), "tau": float(popt[1])}


@register("bi")
def bi(times, counts, acq: Acquisition, detector: Detector | None = None, sample: Sample | None = None) -> dict[str, float] | None:
    loc = rising_edge(counts, keep=4)
    peak = float(np.max(counts))
    guess = (0.6 * peak, 2.0, 0.4 * peak, 8.0)
    bounds = ([0, 1e-8, 0, 1e-8], [np.inf, np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(bi_model, times[loc:], counts[loc:], p0=guess, bounds=bounds, maxfev=8000)
    except (RuntimeError, ValueError):
        return None
    return {"A1": float(popt[0]), "tau1": float(popt[1]), "A2": float(popt[2]), "tau2": float(popt[3])}


@register("mono_conv")
def mono_conv(times, counts, acq: Acquisition, detector: Detector | None = None, sample: Sample | None = None) -> dict[str, float] | None:
    model = lambda t, amp, tau: mono_conv_model(t, amp, tau, detector.irf_mean, detector.irf_width, acq.width)
    try:
        popt, _ = curve_fit(model, times, counts, p0=(float(np.max(counts)), 2.0), maxfev=8000)
    except (RuntimeError, ValueError):
        return None
    return {"A": float(popt[0]), "tau": float(popt[1])}


@register("bi_conv")
def bi_conv(times, counts, acq: Acquisition, detector: Detector | None = None, sample: Sample | None = None) -> dict[str, float] | None:
    peak = float(np.max(counts))
    model = lambda t, a1, tau1, a2, tau2: bi_conv_model(t, a1, tau1, a2, tau2, detector.irf_mean, detector.irf_width, acq.width)
    guess = (0.6 * peak, 2.0, 0.4 * peak, 8.0)
    bounds = ([0, 1e-8, 0, 1e-8], [np.inf, np.inf, np.inf, np.inf])
    try:
        popt, _ = curve_fit(model, times, counts, p0=guess, bounds=bounds, maxfev=8000)
    except (RuntimeError, ValueError):
        return None
    return {"A1": float(popt[0]), "tau1": float(popt[1]), "A2": float(popt[2]), "tau2": float(popt[3])}
