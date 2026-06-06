"""Analysis algorithms. Importing this package registers the built-in fit methods."""
from . import exp, rld  # noqa: F401  (import for @register side effects)
from .correct import correct_bit_depth, max_counts
from .fitter import Fitter, build_maps, sort_components
from .registry import available, fit_funcs, register

__all__ = [
    "Fitter",
    "available",
    "fit_funcs",
    "register",
    "correct_bit_depth",
    "max_counts",
    "build_maps",
    "sort_components",
]
