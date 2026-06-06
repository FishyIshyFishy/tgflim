"""Name -> fitting-function registry.

Every fit method shares the signature ``(times, counts, acq, detector=None,
sample=None) -> dict | None`` so methods are interchangeable and their result dicts
feed map-building directly. New algorithms (Bayesian, phasor, TCSPC) register here.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..params import Acquisition, Detector, Sample

FitFunc = Callable[..., "dict[str, float] | None"]

fit_funcs: dict[str, FitFunc] = {}


def register(name: str) -> Callable[[FitFunc], FitFunc]:
    def decorator(func: FitFunc) -> FitFunc:
        fit_funcs[name] = func
        return func

    return decorator


def available() -> list[str]:
    return sorted(fit_funcs)


def get(name: str) -> FitFunc:
    if name not in fit_funcs:
        raise KeyError(f"unknown fit method {name!r}; choose from {available()}")
    return fit_funcs[name]
