"""Deterministic time-gated forward model: decay -> per-gate detection probability.

The photon arrival intensity of one exponential component, normalised so that it
integrates to one over all time, is ``(1/tau) * exp(-t/tau)`` for ``t >= 0``. The
per-gate detection probability is the integral of this intensity over a gate window
``[t, t+width]``; for an ideal detector that integral is the familiar exponential
difference ``exp(-t/tau) - exp(-(t+width)/tau)``.

The instrument response in *time-gated* FLIM is uncertainty in the pulse/gate origin
time, ``t0 ~ Normal(irf_mean, irf_width)``. The measured signal averages the gate
integral over ``t0``, which is exactly convolving the *causal* decay with the Gaussian
origin-jitter distribution **before** integrating over the gate. That convolution has
the exponentially-modified-Gaussian (EMG) closed form ``emg_kernel`` below, so we
evaluate the convolved intensity analytically and then integrate it over each gate
numerically. This differs from convolving the already-gated discrete trace, which
mishandles causality and the trace edges.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.special import erfc

if TYPE_CHECKING:
    from ..params import Detector, Sample


def emg_kernel(t: np.ndarray, tau: float, irf_mean: float, irf_width: float) -> np.ndarray:
    """Causal exponential intensity convolved with a Gaussian origin jitter.

    Returns the EMG, normalised to unit area (the same erfc expression used by the
    IRF-convolved fits, so simulation and fitting share one kernel).
    """
    lam = 1.0 / tau
    arg_exp = 0.5 * lam * (2.0 * irf_mean + lam * irf_width**2 - 2.0 * t)
    arg_erfc = (irf_mean + lam * irf_width**2 - t) / (irf_width * np.sqrt(2.0))
    return 0.5 * lam * np.exp(arg_exp) * erfc(arg_erfc)


def decay_rate(t: np.ndarray, sample: Sample, detector: Detector) -> np.ndarray:
    """Photon arrival intensity at time(s) ``t`` for the full multi-exponential sample."""
    rate = np.zeros_like(np.asarray(t, dtype=float))
    for tau, weight in zip(sample.lifetimes, sample.weights):
        if detector.irf_width > 0:
            component = emg_kernel(t, tau=tau, irf_mean=detector.irf_mean, irf_width=detector.irf_width)
        else:
            component = (1.0 / tau) * np.exp(-t / tau) * (t >= 0)
        rate += weight * sample.zeta * component
    return rate


def gate_integral(times: np.ndarray, width: float, rate_fn: Callable[[np.ndarray], np.ndarray], substeps: int) -> np.ndarray:
    """Integrate an intensity function over each gate window ``[t, t+width]``."""
    offsets = np.linspace(0.0, width, substeps + 1)
    windows = times[:, None] + offsets[None, :]
    return np.trapezoid(rate_fn(windows), offsets, axis=1)


def gated_prob(times: np.ndarray, sample: Sample, detector: Detector, width: float, substeps: int = 16) -> np.ndarray:
    """Per-gate detection probability for each gate opening in ``times``."""
    if detector.irf_width > 0:
        prob = gate_integral(times, width=width, rate_fn=lambda u: decay_rate(u, sample=sample, detector=detector), substeps=substeps)
    else:
        prob = np.zeros_like(times)
        for tau, weight in zip(sample.lifetimes, sample.weights):
            prob += weight * sample.zeta * (np.exp(-times / tau) - np.exp(-(times + width) / tau))
    return np.clip(prob, 0.0, 1.0)
