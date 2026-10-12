"""Strict signal ingestion: samples x channels, volts, common sampling clock."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Recording:
    data: np.ndarray
    fs: float
    channels: list[str]


def load_recording(path, channels, fs=None, unit=None, time_column=None):
    path = Path(path)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("Specify unique channel names in a fixed order")
    ext = path.suffix.lower()
    if ext in {".edf", ".bdf", ".fif", ".set"}:
        import mne
        readers = {".edf": mne.io.read_raw_edf, ".bdf": mne.io.read_raw_bdf,
                   ".fif": mne.io.read_raw_fif, ".set": mne.io.read_raw_eeglab}
        raw = readers[ext](str(path), preload=False, verbose="ERROR")
        if any(ch not in raw.ch_names for ch in channels):
            raise ValueError(f"Missing selected channels; available: {raw.ch_names}")
        # MNE returns EEG/EOG/EMG channels in SI units. Other sensor types are rejected.
        types = dict(zip(raw.ch_names, raw.get_channel_types()))
        if any(types[ch] not in {"eeg", "eog", "emg"} for ch in channels):
            raise ValueError("Only EEG/EOG/EMG voltage channels are supported")
        sample_rate = float(raw.info["sfreq"])
        data = raw.get_data(picks=channels).T
        raw.close()
    elif ext == ".npz":
        with np.load(path, allow_pickle=False) as z:
            names = z["channels"].astype(str).tolist()
            data = z["data"][:, [names.index(ch) for ch in channels]]
            sample_rate = float(z["fs"])
            stored_unit = str(z["unit"].item())
        if stored_unit not in {"V", "uV"}:
            raise ValueError("NPZ unit must be V or uV")
        data = data * (1e-6 if stored_unit == "uV" else 1.0)
    elif ext in {".csv", ".tsv", ".txt", ".npy"}:
        if unit not in {"V", "uV"}:
            raise ValueError("CSV/NPY require explicit unit=V or uV")
        if ext == ".npy":
            data = np.load(path, allow_pickle=False)
            if data.ndim == 1:
                data = data[:, None]
            if data.ndim != 2 or data.shape[1] != len(channels):
                raise ValueError("NPY must have samples x selected channels; no automatic transpose")
        else:
            frame = pd.read_csv(path, sep="\t" if ext == ".tsv" else None, engine="python")
            data = frame[channels].apply(pd.to_numeric, errors="raise").to_numpy()
            if time_column:
                times = frame[time_column].to_numpy(dtype=float)
                dt = np.diff(times)
                if not len(dt) or not np.isfinite(times).all() or np.any(dt <= 0):
                    raise ValueError("Time must be finite, increasing, and expressed in seconds")
                if not np.allclose(dt, np.median(dt), rtol=1e-3, atol=1e-8):
                    raise ValueError("Irregular/missing timestamps: resample explicitly before import")
                inferred = 1 / np.median(dt)
                if fs is not None and not np.isclose(fs, inferred, rtol=1e-3):
                    raise ValueError("Declared fs disagrees with time column")
                fs = inferred
        if fs is None:
            raise ValueError("Sampling frequency is required; no 256 Hz assumption")
        sample_rate = float(fs)
        data = data * (1e-6 if unit == "uV" else 1.0)
    else:
        raise ValueError(f"Unsupported format {ext}")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("Signals must be finite samples x channels; gaps are never silently dropped")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Invalid sampling frequency")
    return Recording(data, sample_rate, list(channels))


def load_stages(path, offset_s=0.0):
    """Intervals in seconds from signal start; explicit offset supports separate EDF clocks."""
    path = Path(path)
    if path.suffix.lower() == ".edf":
        import mne
        a = mne.read_annotations(str(path))
        frame = pd.DataFrame({"onset": a.onset, "duration": a.duration, "stage": a.description})
    else:
        frame = pd.read_csv(path)
    frame = frame[["onset", "duration", "stage"]].copy()
    frame["onset"] = pd.to_numeric(frame.onset) + float(offset_s)
    frame["duration"] = pd.to_numeric(frame.duration)
    aliases = {"Sleep stage W": "W", "Sleep stage R": "REM", "R": "REM",
               "Sleep stage 1": "N1", "Sleep stage 2": "N2", "Sleep stage 3": "N3",
               "Sleep stage 4": "N3", "1": "N1", "2": "N2", "3": "N3", "4": "N3"}
    frame["stage"] = frame.stage.astype(str).replace(aliases)
    frame = frame.sort_values("onset").reset_index(drop=True)
    if not len(frame) or not np.isfinite(frame[["onset", "duration"]]).all().all():
        raise ValueError("Empty or nonfinite annotations")
    if (frame.duration <= 0).any() or (frame.onset < 0).any():
        raise ValueError("Annotations need positive duration and nonnegative onset")
    if np.any(frame.onset.to_numpy()[1:] < (frame.onset + frame.duration).to_numpy()[:-1] - 1e-6):
        raise ValueError("Overlapping annotations")
    return frame
