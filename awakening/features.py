"""Past-only signal features. No annotation or future sample enters this module."""
import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist, squareform

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "low_gamma": (30, 45)}


def filter_causal(data, fs, cfg):
    if cfg.high_hz >= fs / 2:
        raise ValueError("high_hz must be strictly below Nyquist; cuts are never silently clipped")
    sos = signal.butter(cfg.filter_order, [cfg.low_hz, cfg.high_hz],
                        btype="bandpass", fs=fs, output="sos")
    # Zero initial conditions and explicit discarded warm-up; identical to a streaming pass.
    return signal.sosfilt(sos, np.asarray(data), axis=0)


def delay_embedding(data, tau, dimension):
    """Rows end at their newest sample; columns are [all current, all past, ...]."""
    a = np.asarray(data)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or tau < 1 or dimension < 2:
        raise ValueError("Require samples x channels, tau >= 1 and dimension >= 2")
    count = len(a) - (dimension - 1) * tau
    if count < 2:
        raise ValueError("Not enough samples for embedding")
    latest = np.arange((dimension - 1) * tau, len(a))
    return np.concatenate([a[latest - k * tau] for k in range(dimension)], axis=1)


def geometry(data, fs, cfg):
    if np.any(data.std(axis=0) < 1e-15):
        raise ValueError("Flat channel in window; investigate signal quality before analysis")
    # Window-local centering/scaling uses only past samples; absolute amplitude is retained separately.
    a = (data - data.mean(axis=0)) / np.maximum(data.std(axis=0), 1e-12)
    tau = max(1, int(round(cfg.tau_s * fs)))
    E = delay_embedding(a, tau, cfg.dimension)
    indices = np.unique(np.linspace(0, len(E) - 1, min(cfg.max_points, len(E))).astype(int))
    E = E[indices]
    centered = E - E.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular ** 2
    fractions = variance / max(variance.sum(), 1e-12)
    effective_dim = float(1 / max(np.sum(fractions ** 2), 1e-12))
    # Distance per coordinate, so joint dimensionality does not trivially inflate the radius.
    distances = squareform(pdist(E)) / np.sqrt(E.shape[1])
    valid = np.abs(indices[:, None] - indices[None, :]) > cfg.theiler_s * fs
    upper = valid & np.triu(np.ones_like(valid), 1).astype(bool)
    pairs = distances[upper]
    if not len(pairs):
        raise ValueError("No pairs outside the Theiler window")
    recurrence = (distances < cfg.recurrence_radius) & valid
    # Conditional consecutive recurrences, not the full RQA determinism statistic.
    transitions = recurrence[:-1, :-1] & recurrence[1:, 1:]
    persistence = float(transitions.sum() / max(recurrence[:-1, :-1].sum(), 1))
    return {"effective_rank": effective_dim, "first_pc_fraction": float(fractions[0]),
            "distance_median": float(np.median(pairs)),
            "recurrence_rate": float(np.mean(pairs < cfg.recurrence_radius)),
            "recurrence_persistence": persistence}


def extract_window(data, fs, channels, cfg):
    if data.ndim != 2 or data.shape[1] != len(channels) or not np.isfinite(data).all():
        raise ValueError("Invalid window")
    out = {}
    integrate = getattr(np, "trapezoid", None) or np.trapz
    for c, name in enumerate(channels):
        x = data[:, c]
        freqs, psd = signal.welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
        absolute = {}
        for band, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < min(hi, cfg.high_hz + 1e-9))
            absolute[band] = float(integrate(psd[mask], freqs[mask])) if mask.sum() >= 2 else 0.0
        total = sum(absolute.values())
        for band, value in absolute.items():
            out[f"spectral__{name}__{band}"] = value / max(total, 1e-30)
        out[f"spectral__{name}__log_rms_uv"] = float(np.log10(max(np.sqrt(np.mean(x*x)) * 1e6, 1e-9)))
        for k, value in geometry(data[:, c:c+1], fs, cfg).items():
            out[f"individual__{name}__{k}"] = value
    for k, value in geometry(data, fs, cfg).items():
        out[f"joint__{k}"] = value
    if not np.isfinite(list(out.values())).all():
        raise ValueError("Nonfinite features")
    return out


def feature_rows(recording, cfg):
    filtered = filter_causal(recording.data, recording.fs, cfg)
    fs = recording.fs
    if not all(np.isclose(s*fs, round(s*fs), rtol=0, atol=1e-6) for s in (cfg.window_s, cfg.step_s)):
        raise ValueError("window_s and step_s must span integer sample counts at this fs")
    width = int(round(cfg.window_s * fs))
    step = max(1, int(round(cfg.step_s * fs)))
    start = width + int(np.ceil(cfg.warmup_s * fs))
    previous = None
    for end in range(start, len(filtered) + 1, step):
        features = extract_window(filtered[end-width:end], fs, recording.channels, cfg)
        # Include evolution over the previous decision interval, never a future difference.
        delta = {"delta__" + k: (v - previous[k]) if previous is not None else 0.0
                 for k, v in features.items()}
        previous = features
        yield end / fs, {**features, **delta}
