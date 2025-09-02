# fit/exp.py
import numpy as np
from scipy.optimize import curve_fit

def _mono(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def _bi(t, A, tau1, B, tau2, C):
    return A * np.exp(-t / tau1) + B * np.exp(-t / tau2) + C

def mono(t, trace, **kwargs):
    if np.sum(trace) <= 0:
        return None
    p0 = (float(np.max(trace)), max(t[1]-t[0], 1e-3), 0.0)
    try:
        popt, _ = curve_fit(_mono, t, trace, p0=p0, maxfev=5000)
        A, tau, C = popt
        return {"A": A, "tau": tau, "C": C}
    except Exception:
        return None

def bi(t, trace, **kwargs):
    if np.sum(trace) <= 0:
        return None
    peak = float(np.max(trace))
    # very mild, generic guesses
    p0 = (0.6*peak, max(t[1]-t[0], 1e-3), 0.4*peak, 5*max(t[1]-t[0], 1e-3), 0.0)
    try:
        popt, _ = curve_fit(_bi, t, trace, p0=p0, maxfev=8000)
        A, tau1, B, tau2, C = popt
        # put taus in ascending order for consistency
        if tau1 > tau2:
            A, B = B, A
            tau1, tau2 = tau2, tau1
        return {"A": A, "tau1": tau1, "B": B, "tau2": tau2, "C": C}
    except Exception:
        return None
