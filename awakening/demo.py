"""Synthetic integration fixture with deliberately injected pre-wake signal changes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def make_demo(directory, seed=42):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    fs, duration = 100, 1500
    time = np.arange(fs*duration)/fs
    rng = np.random.default_rng(seed)
    records = []
    for i in range(8):
        first, second = 500+i*7, 1000+i*5
        stages = [(0, first, "N2"), (first, 100, "W"),
                  (first+100, second-first-100, "N2"), (second, 100, "W"),
                  (second+100, duration-second-100, "N2")]
        precursor = sum(np.clip((time-(event-160))/120, 0, 1) * (time < event+100)
                        for event in (first, second))
        base = 20*np.sin(2*np.pi*(2+i*.03)*time)
        data = np.column_stack([base + (3+precursor*20)*np.sin(2*np.pi*12*time)
                                + rng.normal(0, 5, len(time)),
                                0.7*base+(3+precursor*16)*np.sin(2*np.pi*12*time+.4)
                                + rng.normal(0, 6, len(time))])
        signal_path, stages_path = f"s{i:02d}.npz", f"s{i:02d}_stages.csv"
        np.savez_compressed(root/signal_path, data=data, fs=fs, channels=np.array(["EEG1", "EEG2"]), unit="uV")
        pd.DataFrame(stages, columns=["onset", "duration", "stage"]).to_csv(root/stages_path, index=False)
        records.append({"id": f"s{i:02d}_night1", "subject": f"s{i:02d}",
                        "split": "train" if i < 4 else ("validation" if i < 6 else "test"),
                        "signal": signal_path, "annotations": stages_path})
    path = root/"manifest.json"
    path.write_text(json.dumps({"synthetic": True, "channels": ["EEG1", "EEG2"], "recordings": records}, indent=2))
    return path
