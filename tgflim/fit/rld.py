# fit/rld.py
import numpy as np

def mono_rld(t, trace, **kwargs):
    """Two-gate RLD for monoexp. Uses the first 2 gates of the trace."""
    if len(trace) < 2:
        return None
    D0, D1 = float(trace[0]), float(trace[1])
    if D0 <= 0 or D1 <= 0 or D0 == D1:
        return None
    step = kwargs.get("step", t[1]-t[0])
    # classic formulas
    tau = step / np.log(D0 / D1)
    if not np.isfinite(tau) or tau <= 0:
        return None
    # crude amplitude estimate at t=0 (offset ignored)
    A = D0 - 0.0
    return {"A": A, "tau": tau, "C": 0.0}

def bi_rld(t, trace, **kwargs):
    if len(trace) < 4:
        return None

    D0, D1, D2, D3 = [float(x) for x in trace[:4]]
    if any(v <= 0 for v in (D0, D1, D2, D3)):
        return None
    step = kwargs.get("step", t[1]-t[0])

    try:
        tau1 = step / np.log(D0 / D1)
        tau2 = step / np.log(D2 / D3)
    except Exception:
        return None
    if not (np.isfinite(tau1) and np.isfinite(tau2)) or tau1 <= 0 or tau2 <= 0:
        return None
    if tau1 > tau2:
        tau1, tau2 = tau2, tau1

    A = max(D0 - 0.0, 0.0)
    B = max(D2 - 0.0, 0.0)
    return {"A": A, "tau1": tau1, "B": B, "tau2": tau2, "C": 0.0}
