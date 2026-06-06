"""Decay-trace plotting with an optional fitted-model overlay."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..fit.exp import bi_model, bi_conv_model, mono_model, mono_conv_model

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..core import PixelTrace
    from ..params import Acquisition, Detector


def model_curve(times: np.ndarray, result: dict[str, float], method: str, detector: Detector | None, width: float) -> np.ndarray | None:
    if method in ("mono", "mono_rld", "mono_rld_50ovp"):
        return mono_model(times, result["A"], result["tau"])
    if method == "mono_conv":
        return mono_conv_model(times, result["A"], result["tau"], detector.irf_mean, detector.irf_width, width)
    if method in ("bi", "bi_rld"):
        return bi_model(times, result["A1"], result["tau1"], result["A2"], result["tau2"])
    if method == "bi_conv":
        return bi_conv_model(times, result["A1"], result["tau1"], result["A2"], result["tau2"], detector.irf_mean, detector.irf_width, width)
    return None


def plot_trace_fit(ax: Axes, trace: PixelTrace, result: dict[str, float] | None = None, method: str | None = None, detector: Detector | None = None, acq: Acquisition | None = None) -> None:
    ax.plot(trace.times, trace.counts, "o", ms=3, color="0.3", label="data")
    if result and method:
        curve = model_curve(trace.times, result, method, detector, acq.width if acq else 0.0)
        if curve is not None:
            label = fit_label(result)
            ax.plot(trace.times, curve, "-", color="crimson", label=label)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("counts")
    ax.legend()


def fit_label(result: dict[str, float]) -> str:
    if "tau1" in result:
        return f"fit: tau1={result['tau1']:.2f}, tau2={result['tau2']:.2f} ns"
    return f"fit: tau={result['tau']:.2f} ns"
