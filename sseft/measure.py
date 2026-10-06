"""
Measurement utilities of the programme: the Phase 1c fibre measurement (ensemble spread, local exponents,
closure) and the X2 regulator level.
"""
import numpy as np


def ensemble_stats(preds):
    """preds (K, n): mean and std over the ensemble."""
    preds = np.asarray(preds)
    return preds.mean(axis=0), preds.std(axis=0, ddof=1)


def local_exponent(x, y):
    """-d log y / d log x by centred differences (nan at the ends and where y <= 0)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lx, ly = np.log(x), np.log(np.where(y > 0, y, np.nan))
    q = np.full(len(x), np.nan)
    q[1:-1] = -(ly[2:] - ly[:-2]) / (lx[2:] - lx[:-2])
    return q


def closure(R, W, f_true, p_star=6.0):
    """slope of log(W/|f*|) vs log R should be p* - q; returns the relative spread and its local slope."""
    rel = np.asarray(W) / np.abs(np.asarray(f_true))
    return rel, -local_exponent(R, rel)


def plateau_slope(x, y, lo, hi):
    """least-squares slope of log y vs log x on [lo, hi]."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = (x >= lo) & (x <= hi) & (y > 0)
    if m.sum() < 3: return np.nan
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]


def level(resid_a, resid_b, ref):
    """X2 level: rms of the regulator difference of the residuals, relative to rms of the reference."""
    resid_a = np.asarray(resid_a); resid_b = np.asarray(resid_b); ref = np.asarray(ref)
    return np.sqrt(np.mean((resid_a - resid_b) ** 2)) / np.sqrt(np.mean(ref ** 2))
