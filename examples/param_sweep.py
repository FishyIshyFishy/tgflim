"""Monte-Carlo sweep: how does fitted-lifetime precision improve with integration time?

This is the figure pattern the package is built for -- the sweep loop lives here in the
script, not in the package.
"""
from __future__ import annotations

import dataclasses

import matplotlib.pyplot as plt
import numpy as np

from tgflim import Acquisition, Detector, Fitter, Generator, Sample, TimeGated

base = Acquisition(freq=10, width=5.0, step=0.5, offset=2.5, bits=8, integ=10000)
sample = Sample(lifetimes=[20.0], weights=[1.0], zeta=0.05)
detector = Detector(irf_width=0.2, dark_cps=25)

integs = np.array([1000, 2500, 5000, 10000, 20000])
realizations = 200
means = np.zeros(len(integs))
stds = np.zeros(len(integs))

for k, integ in enumerate(integs):
    acq = dataclasses.replace(base, integ=float(integ))
    gen = Generator(acq, TimeGated(), seed=k)
    fitter = Fitter(acq, "mono", correct="antol")
    taus = []
    for _ in range(realizations):
        result = fitter.fit_trace(gen.trace(sample, detector))
        if result:
            taus.append(result["tau"])
    taus = np.array(taus)
    means[k], stds[k] = taus.mean(), taus.std()

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(integs / 1e3, means, yerr=stds, fmt="o-", capsize=4)
ax.axhline(20.0, ls="--", color="gray", label="ground truth")
ax.set_xlabel("integration time (ms)")
ax.set_ylabel("fitted lifetime (ns)")
ax.legend()
plt.tight_layout()
plt.show()
