"""Saving and loading simulated cubes and fit-result maps (npz). No hard-coded paths."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_cube(path: str | Path, cube: np.ndarray, times: np.ndarray) -> None:
    np.savez(Path(path).with_suffix(".npz"), cube=cube, times=times)


def load_cube(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(Path(path).with_suffix(".npz"))
    return data["cube"], data["times"]


def save_result_maps(path: str | Path, maps: dict[str, np.ndarray]) -> None:
    np.savez(Path(path).with_suffix(".npz"), **maps)


def load_result_maps(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path).with_suffix(".npz"))
    return {k: data[k] for k in data.files}
