import json
import platform
from pathlib import Path
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import Config
from .dataset import build_dataset
from .evaluation import (choose_threshold, event_metrics, subject_bootstrap, window_metrics,
                         reliability_table, paired_subject_difference)
from .features import feature_rows
from .io import load_recording

MODELS = ("prevalence", "spectral", "individual", "joint")


def columns_for(frame, kind):
    families = {"prevalence": ("spectral__",), "spectral": ("spectral__",),
                "individual": ("spectral__", "individual__"), "joint": ("spectral__", "joint__")}[kind]
    return [c for c in frame if c.startswith(families) or
            (c.startswith("delta__") and c[len("delta__"):].startswith(families))]


def run_experiment(manifest, cfg, output, false_alarm_budget=0.5):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    frame, events, provenance = build_dataset(manifest, cfg)
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("All three subject-disjoint splits are required")
    train = frame[frame.split == "train"]
    if train.label.nunique() != 2:
        raise ValueError("Training needs both positive and negative windows")
    frame.to_csv(out / "features.csv", index=False)
    events.to_csv(out / "events.csv", index=False)
    versions = {p: importlib.metadata.version(p) for p in
                ["numpy", "scipy", "pandas", "scikit-learn", "joblib"]}
    report = {"config": cfg.to_dict(), "provenance": provenance, "versions": versions,
              "python": platform.python_version(), "false_alarm_budget_per_hour": false_alarm_budget,
              "target": "Sustained awakening onset in [t+lead, t+lead+horizon)",
              "evaluation_support": "All fully annotated windows, including wake; no oracle sleep gate",
              "model_selection": "All four prespecified models reported; no best-model selection on test",
              "models": {}}
    test_records = {}
    for kind in MODELS:
        columns = columns_for(frame, kind)
        estimator = (DummyClassifier(strategy="prior") if kind == "prevalence" else
                     make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000,
                                                                         random_state=cfg.seed)))
        estimator.fit(train[columns], train.label)
        results = {}
        for split in ("validation", "test"):
            subset = frame[frame.split == split]
            prediction = subset[["recording", "subject", "split", "time_s", "label"]].copy()
            prediction["probability"] = estimator.predict_proba(subset[columns])[:, 1]
            prediction["synthetic_model"] = provenance["synthetic"]
            results[split] = prediction
        threshold, curve = choose_threshold(results["validation"], events, cfg, false_alarm_budget)
        pd.DataFrame(curve).to_csv(out / f"{kind}_validation_thresholds.csv", index=False)
        report["models"][kind] = {"threshold": threshold, "feature_count": len(columns)}
        for split, prediction in results.items():
            stats, records, alerts = event_metrics(prediction, events, threshold, cfg)
            report["models"][kind][split] = {**stats, **window_metrics(prediction),
                                            "bootstrap": subject_bootstrap(records, cfg.seed)}
            prediction.to_csv(out / f"{kind}_{split}_predictions.csv", index=False)
            alerts.to_csv(out / f"{kind}_{split}_alerts.csv", index=False)
            records.to_csv(out / f"{kind}_{split}_recordings.csv", index=False)
            reliability_table(prediction).to_csv(out / f"{kind}_{split}_reliability.csv", index=False)
            if split == "test":
                test_records[kind] = records
        joblib.dump({"format_version": 1, "estimator": estimator, "features": columns,
                     "channels": provenance["channels"], "config": cfg.to_dict(),
                     "threshold": threshold, "model_kind": kind, "versions": versions,
                     "supported_fs": sorted({r["fs"] for r in provenance["recordings"] if r["split"] == "train"}),
                     "synthetic_training": provenance["synthetic"]}, out / f"{kind}.joblib")
    report["primary_comparison"] = paired_subject_difference(test_records["spectral"], test_records["joint"], cfg.seed)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ["# Awakening experiment", "", "SYNTHETIC DEMONSTRATION — not evidence of EEG performance."
             if provenance["synthetic"] else "Held-out results for this dataset and protocol only.", "",
             "| Model | Test events | Sensitivity | False alarms/h | Brier |", "|---|---:|---:|---:|---:|"]
    for name, values in report["models"].items():
        r = values["test"]
        sensitivity = "undefined" if r["sensitivity"] is None else f"{r['sensitivity']:.3f}"
        lines.append(f"| {name} | {r['events']} | {sensitivity} | {r['false_alarms_per_hour']:.3f} | {r['brier']:.3f} |")
    lines += ["", "Thresholds selected on validation subjects only. The false-alarm constraint is",
              "a validation constraint, not a guarantee on new subjects. See report.json for CIs,",
              "provenance, configuration, sample counts and probabilistic metrics."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return report


def predict_recording(model_path, signal_path, fs=None, unit=None, time_column=None):
    # joblib is executable serialization: caller must only load a trusted, locally trained model.
    bundle = joblib.load(model_path)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported model format")
    cfg = Config(**bundle["config"])
    rec = load_recording(signal_path, bundle["channels"], fs, unit, time_column)
    if not any(np.isclose(rec.fs, rate, rtol=1e-6) for rate in bundle["supported_fs"]):
        raise ValueError("Sampling rate absent from training; resample with a documented causal pipeline or retrain")
    values = [{"time_s": t, **features} for t, features in feature_rows(rec, cfg)]
    if not values:
        raise ValueError("Recording too short for window plus filter warm-up")
    frame = pd.DataFrame(values)
    probabilities = bundle["estimator"].predict_proba(frame[bundle["features"]])[:, 1]
    from .evaluation import issue_alerts
    alerts = issue_alerts(frame.time_s.to_numpy(), probabilities, bundle["threshold"], cfg)
    return pd.DataFrame({"time_s": frame.time_s, "probability": probabilities,
                         "alert": frame.time_s.isin(alerts), "target_start_s": frame.time_s+cfg.lead_s,
                         "target_end_s": frame.time_s+cfg.lead_s+cfg.horizon_s,
                         "synthetic_model": bundle["synthetic_training"]})
