"""Generate a two-region FLIM image and recover a per-pixel lifetime map.

Demonstrates the spatial pipeline: an ``Image`` of ground-truth samples -> a cube ->
per-pixel fitting. ``exec_mode=pool`` fans the per-pixel fits across processes, which on
Windows requires the ``if __name__ == "__main__"`` guard below.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from tgflim import Acquisition, Detector, ExecMode, Fitter, Generator, Image, TimeGated
from tgflim.viz import plot_result_maps


def main() -> None:
    x = y = 24
    acq = Acquisition(freq=10, width=1.0, step=0.25, offset=0.0, bits=12, integ=30000, exec_mode=ExecMode.pool)
    detector = Detector(irf_mean=0.4, irf_width=0.2, dark_cps=25)

    # left half = 4 ns, right half = 10 ns (single component)
    lifetimes = np.full((1, x, y), 4.0)
    lifetimes[0, :, y // 2:] = 10.0
    weights = np.ones((1, x, y))
    zeta = np.full((x, y), 0.02)

    image = Image(acq, detector, lifetimes, weights, zeta_map=zeta)
    cube = Generator(acq, TimeGated(), seed=1).image(image)

    maps = Fitter(acq, "mono", correct="antol", detector=detector).fit_image(cube)
    print("median tau, left region :", np.median(maps["tau"][:, : y // 2]))
    print("median tau, right region:", np.median(maps["tau"][:, y // 2:]))

    fig, _ = plot_result_maps(maps, keys=["tau"])
    fig.suptitle("recovered lifetime map (mono)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
