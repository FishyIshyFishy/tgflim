import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional



@dataclass
class Config:
    # acquisition params
    frames: int = 0
    bits: int = 8
    power: float = 0.0
    freq: float = 20.0           # MHz
    integ: float = 1000.0        # us
    width: float = 0.5           # ns
    step: float = 0.1            # ns
    offset: float = 0.0          # ns
    numsteps: Optional[int] = int((1e3/freq) / step)

    # IRF / noise
    irf_mean: float = 0.0        # ns
    irf_width: float = 0.0       # ns
    dark_cps: float = 0.0

    # decay model
    lifetimes: list[float] = [2.5]
    weight: list[float] = [1.0]
    zeta: float = 1.0

    # image config
    x: int = 100
    y: int = 100
    filename: str = r'a'
    folder: str = r'a'

    def __post_init__(self):
        self.times = (np.arange(self.numsteps) * self.step) + self.offset
