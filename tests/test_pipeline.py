import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from awakening.config import Config
from awakening.features import delay_embedding, filter_causal, feature_rows, geometry
from awakening.io import Recording, load_recording, load_stages
from awakening.labels import awakening_events, covered, future_label
from awakening.dataset import read_manifest
from awakening.evaluation import issue_alerts, event_metrics, choose_threshold
from awakening.diagnostics import phase_surrogate, ami_delay, fnn_curve


def stages(rows):
    return pd.DataFrame(rows, columns=["onset", "duration", "stage"])


def test_embedding_uses_past_and_all_channels():
    a = np.arange(20).reshape(10, 2)
    E = delay_embedding(a, 2, 4)
    np.testing.assert_array_equal(E[0], [12, 13, 8, 9, 4, 5, 0, 1])
    assert E.shape == (4, 8)


def test_embedding_accepts_noncontiguous_input_safely():
    a = np.arange(40).reshape(20, 2)[::2]
    np.testing.assert_array_equal(delay_embedding(a, 1, 3)[0], [8, 9, 4, 5, 0, 1])


def test_causal_filter_unchanged_when_future_is_replaced():
    rng = np.random.default_rng(42)
    a = rng.normal(size=(10000, 2))
    b = a.copy()
    b[6000:] = rng.normal(100, 30, size=b[6000:].shape)
    np.testing.assert_array_equal(filter_causal(a, 100, Config())[:6000],
                                  filter_causal(b, 100, Config())[:6000])


def test_all_features_and_deltas_are_prefix_invariant():
    rng = np.random.default_rng(12)
    cfg = Config(window_s=10, step_s=5, warmup_s=5, max_points=32)
    a = rng.normal(size=(3000, 2)) * 1e-5
    full = list(feature_rows(Recording(a, 100, ["a", "b"]), cfg))
    prefix = list(feature_rows(Recording(a[:2000], 100, ["a", "b"]), cfg))
    assert full[:len(prefix)] == prefix


def test_sustained_wake_merges_epochs_and_requires_contiguous_sleep():
    s = stages([(0, 60, "N2"), (60, 60, "N3"), (120, 30, "W"), (150, 30, "W"),
                (180, 130, "N2"), (310, 20, "W"), (330, 130, "N2"), (470, 80, "W")])
    np.testing.assert_array_equal(awakening_events(s, 120, 60), [120])


def test_unknown_stages_and_gaps_are_not_negative_examples():
    s = stages([(0, 50, "N2"), (50, 20, "?"), (70, 30, "W"), (110, 30, "N2")])
    assert covered(s, 0, 50)
    assert not covered(s, 0, 51)
    assert not covered(s, 90, 120)
    assert not covered(s, 130, 150)


def test_label_has_gap_and_half_open_horizon():
    cfg = Config(lead_s=30, horizon_s=120)
    assert future_label(100, np.array([130]), cfg) == 1
    assert future_label(100, np.array([129]), cfg) == 0
    assert future_label(100, np.array([250]), cfg) == 0


def test_alert_refractory_and_event_matching():
    cfg = Config(lead_s=30, horizon_s=120)
    t = np.arange(0, 400, 10)
    p = np.ones(len(t))
    np.testing.assert_array_equal(issue_alerts(t, p, .5, cfg), [0, 150, 300])
    pred = pd.DataFrame({"recording": "r", "subject": "s", "time_s": t, "probability": p})
    events = pd.DataFrame({"recording": ["r", "r"], "onset_s": [100, 400]})
    metrics, _, alerts = event_metrics(pred, events, .5, cfg)
    assert metrics["events"] == 2 and metrics["detected"] == 2 and metrics["false_alarms"] == 1
    assert len(alerts) == 3
    assert 0 <= metrics["time_in_warning_fraction"] <= 1


def test_late_alert_is_false_and_not_a_prediction():
    cfg = Config()
    t = np.arange(0, 200, 10)
    pred = pd.DataFrame({"recording": "r", "subject": "s", "time_s": t,
                         "probability": (t == 90).astype(float)})
    events = pd.DataFrame({"recording": ["r"], "onset_s": [100]})
    metrics, _, _ = event_metrics(pred, events, .5, cfg)
    assert metrics["detected"] == 0 and metrics["false_alarms"] == 1


def test_no_event_metrics_are_explicitly_undefined():
    pred = pd.DataFrame({"recording": ["r"], "subject": ["s"], "time_s": [100], "probability": [.8]})
    ev = pd.DataFrame(columns=["recording", "onset_s"])
    metrics, _, _ = event_metrics(pred, ev, .5, Config())
    assert metrics["sensitivity"] is None
    with pytest.raises(ValueError, match="Validation"):
        choose_threshold(pred, ev, Config(), .5)


def test_threshold_can_suppress_all_alerts():
    cfg = Config()
    pred = pd.DataFrame({"recording": "r", "subject": "s", "time_s": np.arange(0, 400, 10), "probability": 1.})
    ev = pd.DataFrame({"recording": ["r"], "onset_s": [100]})
    threshold, _ = choose_threshold(pred, ev, cfg, 0)
    assert threshold > 1


def test_csv_requires_units_and_sampling_rate(tmp_path):
    p = tmp_path/"x.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    with pytest.raises(ValueError, match="unit"):
        load_recording(p, ["a", "b"], fs=100)
    with pytest.raises(ValueError, match="Sampling"):
        load_recording(p, ["a", "b"], unit="uV")
    rec = load_recording(p, ["b", "a"], fs=100, unit="uV")
    np.testing.assert_allclose(rec.data[0], [2e-6, 1e-6])


def test_csv_missing_samples_are_rejected(tmp_path):
    p = tmp_path/"x.csv"
    p.write_text("t,a\n0,1\n0.01,2\n0.03,3\n")
    with pytest.raises(ValueError, match="Irregular"):
        load_recording(p, ["a"], unit="V", time_column="t")
    p.write_text("a,b\n1,2\n,4\n")
    with pytest.raises(ValueError, match="finite"):
        load_recording(p, ["a", "b"], fs=100, unit="V")


def test_annotation_overlap_rejected(tmp_path):
    p = tmp_path/"a.csv"
    p.write_text("onset,duration,stage\n0,60,N2\n30,60,W\n")
    with pytest.raises(ValueError, match="Overlapping"):
        load_stages(p)


def test_subject_leakage_rejected_before_training(tmp_path):
    (tmp_path/"a").touch()
    (tmp_path/"b").touch()
    spec = {"channels": ["x"], "recordings": [
        {"id": "1", "subject": "same", "split": "train", "signal": "a", "annotations": "b"},
        {"id": "2", "subject": "same", "split": "test", "signal": "b", "annotations": "a"}]}
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Subject leakage"):
        read_manifest(path)


def test_surrogates_preserve_cross_spectrum():
    x = np.random.default_rng(1).normal(size=(1000, 3))
    a = np.fft.rfft(x, axis=0)
    b = np.fft.rfft(phase_surrogate(x), axis=0)
    np.testing.assert_allclose(abs(a), abs(b), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a[:, 0]*a[:, 1].conj(), b[:, 0]*b[:, 1].conj(), rtol=1e-10, atol=1e-9)


def test_fnn_and_ami_reject_constant_signal():
    with pytest.raises(ValueError):
        ami_delay(np.ones(1000), 30)
    with pytest.raises(ValueError):
        fnn_curve(np.ones(1000), 5)


def test_theiler_excludes_all_pairs_when_too_large():
    x = np.random.default_rng(2).normal(size=(1000, 1))
    with pytest.raises(ValueError, match="No pairs"):
        geometry(x, 100, Config(theiler_s=20))


def test_invalid_nyquist_rejected():
    with pytest.raises(ValueError, match="Nyquist"):
        filter_causal(np.zeros((100, 2)), 50, Config())


def test_constant_geometry_is_not_reported_as_huge_dimension():
    with pytest.raises(ValueError, match="Flat channel"):
        geometry(np.zeros((1000, 2)), 100, Config())


def test_configuration_rejects_fractional_dimension():
    with pytest.raises(ValueError, match="integer"):
        Config(dimension=3.5)
