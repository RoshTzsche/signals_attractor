import json
import numpy as np
import pandas as pd
from awakening.config import Config
from awakening.model import run_experiment, predict_recording


def test_full_training_inference_and_censoring(tmp_path):
    fs, seconds = 100, 180
    t = np.arange(fs*seconds)/fs
    records = []
    for i, split in enumerate(["train", "train", "validation", "validation", "test", "test"]):
        rng = np.random.default_rng(i)
        x = np.column_stack([np.sin(t*2*np.pi*3), np.cos(t*2*np.pi*5)])*1e-5
        x += rng.normal(0, 1e-6, size=x.shape)
        np.savez(tmp_path/f"s{i}.npz", data=x, channels=np.array(["a", "b"]), unit="V", fs=fs)
        pd.DataFrame([(0, 80, "N2"), (80, 30, "W"), (110, 70, "N2")],
                     columns=["onset", "duration", "stage"]).to_csv(tmp_path/f"s{i}.csv", index=False)
        records.append({"id": f"r{i}", "subject": f"s{i}", "split": split,
                        "signal": f"s{i}.npz", "annotations": f"s{i}.csv"})
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"channels": ["a", "b"], "recordings": records, "synthetic": True}))
    cfg = Config(window_s=10, step_s=10, warmup_s=5, lead_s=10, horizon_s=30,
                 min_sleep_s=30, min_wake_s=20, max_points=24)
    report = run_experiment(manifest, cfg, tmp_path/"out")
    features = pd.read_csv(tmp_path/"out/features.csv")
    assert features.time_s.max()+cfg.lead_s+cfg.horizon_s+cfg.min_wake_s <= seconds
    assert features.groupby("subject").split.nunique().max() == 1
    assert set(report["models"]) == {"prevalence", "spectral", "individual", "joint"}
    assert report["models"]["joint"]["test"]["events"] == 2
    predictions = predict_recording(tmp_path/"out/joint.joblib", tmp_path/"s4.npz")
    held = pd.read_csv(tmp_path/"out/joint_test_predictions.csv")
    held = held[held.recording == "r4"]
    common = held.merge(predictions, on="time_s", suffixes=("_test", "_live"))
    np.testing.assert_allclose(common.probability_test, common.probability_live, atol=1e-12)
    assert predictions.time_s.max() > held.time_s.max()  # No future labels needed at inference.
    assert predictions.synthetic_model.all()
    assert (tmp_path/"out/joint_test_reliability.csv").exists()
