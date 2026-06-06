"""Generate one bi-exponential time-gated trace and recover its lifetimes.

Shows the standard pipeline: simulate -> bit-depth correct -> IRF-aware fit.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from tgflim import Acquisition, Detector, Fitter, Generator, Sample, TimeGated
from tgflim.viz import plot_trace_fit

acq = Acquisition(freq=10, width=1.0, step=0.2, offset=0.0, bits=12, integ=30000)
sample = Sample(lifetimes=[3.0, 15.0], weights=[0.5, 0.5], zeta=0.02)
detector = Detector(irf_mean=0.4, irf_width=0.2, dark_cps=25)

trace = Generator(acq, TimeGated(), seed=0).trace(sample, detector)

result = Fitter(acq, "bi_conv", correct="antol", detector=detector).fit_trace(trace)
print("ground truth lifetimes:", sample.lifetimes)
print("bi_conv recovered     :", result)

fig, ax = plt.subplots(figsize=(6, 4))
plot_trace_fit(ax, trace, result, method="bi_conv", detector=detector, acq=acq)
ax.set_title("time-gated bi-exponential decay")
plt.tight_layout()
plt.show()
