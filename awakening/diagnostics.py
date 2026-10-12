"""Exploratory diagnostics, not certificates of an attractor or deterministic chaos."""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .features import delay_embedding


def ami_delay(x, max_lag, bins=16):
    x = np.asarray(x, dtype=float)
    max_lag = min(int(max_lag), len(x)//4)
    if max_lag < 3 or not np.isfinite(x).all() or np.std(x) < 1e-15:
        raise ValueError("Need a finite nonconstant signal and at least three candidate lags")
    values = []
    for tau in range(1, max_lag+1):
        # Deterministic distributed pairs, not merely the first second of the recording.
        indices = np.unique(np.linspace(0, len(x)-tau-1, min(5000, len(x)-tau)).astype(int))
        counts, _, _ = np.histogram2d(x[indices], x[indices+tau], bins=bins)
        joint = counts/counts.sum()
        independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
        mask = joint > 0
        values.append(float(np.sum(joint[mask]*np.log2(joint[mask]/independent[mask]))))
    minima = [i for i in range(1, len(values)-1) if values[i] < values[i-1] and values[i] < values[i+1]]
    index = minima[0] if minima else int(np.argmin(values))
    return index+1, values


def fnn_curve(x, tau, max_dimension=8, theiler=100, max_points=600):
    x = np.asarray(x, dtype=float)
    scale = np.std(x)
    if scale < 1e-15:
        raise ValueError("FNN undefined for constant signals")
    result = []
    for dimension in range(1, max_dimension+1):
        # All candidate points have the extra coordinate: denominator counts only tested pairs.
        extended = delay_embedding(x, tau, dimension+1)
        ids = np.unique(np.linspace(0, len(extended)-1, min(len(extended), max_points)).astype(int))
        extended = extended[ids]
        current = extended[:, :dimension]
        distances = squareform(pdist(current))
        distances[np.abs(ids[:, None]-ids[None, :]) <= theiler] = np.inf
        neighbours = distances.argmin(axis=1)
        nearest = distances[np.arange(len(ids)), neighbours]
        valid = np.isfinite(nearest) & (nearest > 1e-12)
        extra = np.abs(extended[:, -1]-extended[neighbours, -1])
        false = (extra[valid]/nearest[valid] > 15) | (np.hypot(nearest[valid], extra[valid])/scale > 2)
        result.append({"dimension": dimension, "tested_pairs": int(valid.sum()),
                       "false_fraction": float(false.mean()) if valid.any() else None})
    return result


def correlation_curve(E, theiler=100, max_points=600):
    ids = np.unique(np.linspace(0, len(E)-1, min(len(E), max_points)).astype(int))
    E = np.asarray(E)[ids]
    E = (E-E.mean(axis=0))/np.maximum(E.std(axis=0), 1e-12)
    distance = squareform(pdist(E))/np.sqrt(E.shape[1])
    valid = np.triu(np.abs(ids[:, None]-ids[None, :]) > theiler, 1)
    pairs = distance[valid]
    pairs = pairs[np.isfinite(pairs)]
    positive = pairs[pairs > 0]
    if len(positive) < 20:
        return {"r": [], "C": [], "local_slope": [], "note": "Insufficient nonzero pairs"}
    lo, hi = np.quantile(positive, [.02, .8])
    if hi <= lo:
        return {"r": [], "C": [], "local_slope": [], "note": "No distance range"}
    radii = np.geomspace(lo, hi, 20)
    corr = np.array([(pairs < r).mean() for r in radii])
    slopes = np.gradient(np.log(np.maximum(corr, 1e-12)), np.log(radii))
    return {"r": radii.tolist(), "C": corr.tolist(), "local_slope": slopes.tolist(),
            "note": "Local slopes only; a scaling plateau and embedding stability must be examined"}


def phase_surrogate(data, seed=42):
    """Common random Fourier phases preserve auto/cross spectra, not amplitude histograms.

    A stationary linear spectral null. Nonstationarity can also cause rejection;
    this is not a test establishing an attractor or causality.
    """
    a = np.asarray(data)
    if a.ndim == 1:
        a = a[:, None]
    fft = np.fft.rfft(a, axis=0)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2*np.pi, fft.shape[0])
    phase[0] = 0
    if len(a) % 2 == 0:
        phase[-1] = 0
    return np.fft.irfft(fft*np.exp(1j*phase[:, None]), n=len(a), axis=0)
