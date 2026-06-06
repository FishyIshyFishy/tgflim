"""tgflim: modular simulation and analysis for time-gated FLIM.

Quick start
-----------
>>> from tgflim import Acquisition, Sample, Detector, Generator, TimeGated, Fitter
>>> acq = Acquisition(freq=10, width=5.0, step=0.5, offset=2.5, bits=8, integ=10000)
>>> sample = Sample(lifetimes=[5.0, 20.0], weights=[0.5, 0.5], zeta=0.05)
>>> detector = Detector(irf_width=0.2, dark_cps=25)
>>> trace = Generator(acq, TimeGated(), seed=0).trace(sample, detector)
>>> Fitter(acq, "bi", correct="antol").fit_trace(trace)
"""
from __future__ import annotations

from . import fit, forward, grid, io, viz
from .core import Image, PixelTrace
from .fit import Fitter, available
from .forward import AcquisitionModel, TimeGated
from .params import Acquisition, Detector, ExecMode, Sample
from .sim import Generator

__all__ = [
    "Acquisition",
    "Sample",
    "Detector",
    "ExecMode",
    "PixelTrace",
    "Image",
    "AcquisitionModel",
    "TimeGated",
    "Generator",
    "Fitter",
    "available",
    "fit",
    "forward",
    "viz",
    "grid",
    "io",
]
