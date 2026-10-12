"""Fixed event-alarm protocol shared by validation and test."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def issue_alerts(times, probabilities, threshold, cfg):
    """One alert per lead+horizon refractory interval; sustained high risk can re-alert."""
    alerts = []
    next_allowed = -np.inf
    for t, p in zip(times, probabilities):
        if t >= next_allowed and p >= threshold:
            alerts.append(float(t))
            next_allowed = t + cfg.lead_s + cfg.horizon_s
    return np.asarray(alerts)


def event_metrics(predictions, events, threshold, cfg):
    records, alarm_rows = [], []
    for rid, part in predictions.groupby("recording", sort=False):
        part = part.sort_values("time_s")
        times = part.time_s.to_numpy()
        ev = events.loc[events.recording == rid, "onset_s"].to_numpy(dtype=float)
        # Only count events with at least one available decision at a valid advance time.
        ev = np.asarray([e for e in ev if np.any((e >= times + cfg.lead_s) &
                                               (e < times + cfg.lead_s + cfg.horizon_s))])
        alerts = issue_alerts(times, part.probability.to_numpy(), threshold, cfg)
        detected, leads, false = set(), [], 0
        for t in alerts:
            hits = ev[(ev >= t + cfg.lead_s) & (ev < t + cfg.lead_s + cfg.horizon_s)]
            if not len(hits):
                false += 1
            for e in hits:
                if e not in detected:
                    detected.add(e)
                    leads.append(float(e-t))
            alarm_rows.append({"recording": rid, "subject": part.subject.iloc[0], "issued_s": t,
                               "target_start_s": t+cfg.lead_s,
                               "target_end_s": t+cfg.lead_s+cfg.horizon_s,
                               "hit": bool(len(hits))})
        # Time denominator is scored support, not full night (unknown/censored intervals excluded).
        warning = np.zeros(len(times), dtype=float)
        for t in alerts:
            warning += np.maximum(0, np.minimum(times+cfg.step_s, t+cfg.lead_s+cfg.horizon_s)
                                  - np.maximum(times, t+cfg.lead_s))
        records.append({"recording": rid, "subject": part.subject.iloc[0], "events": len(ev),
                        "detected": len(detected), "false_alarms": false,
                        "hours": len(times)*cfg.step_s/3600,
                        "warning_s": float(np.minimum(warning, cfg.step_s).sum()),
                        "lead_times_s": leads})
    table = pd.DataFrame(records)
    if not len(table):
        raise ValueError("No predictions to evaluate")
    total_events, hours = int(table.events.sum()), float(table.hours.sum())
    leads = [v for values in table.lead_times_s for v in values]
    summary = {"events": total_events, "detected": int(table.detected.sum()),
               "sensitivity": float(table.detected.sum()/total_events) if total_events else None,
               "false_alarms": int(table.false_alarms.sum()), "scored_hours": hours,
               "false_alarms_per_hour": float(table.false_alarms.sum()/hours),
               "time_in_warning_fraction": float(table.warning_s.sum()/(hours*3600)),
               "median_lead_s": float(np.median(leads)) if leads else None}
    return summary, table, pd.DataFrame(alarm_rows, columns=["recording", "subject", "issued_s",
                                                          "target_start_s", "target_end_s", "hit"])


def window_metrics(predictions):
    y, p = predictions.label.to_numpy(), predictions.probability.to_numpy()
    both = len(np.unique(y)) == 2
    return {"windows": len(y), "prevalence": float(y.mean()),
            "brier": float(brier_score_loss(y, p)),
            "average_precision": float(average_precision_score(y, p)) if both else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None}


def choose_threshold(predictions, events, cfg, max_false_per_hour):
    if not np.isfinite(max_false_per_hour) or max_false_per_hour < 0:
        raise ValueError("False-alarm budget must be nonnegative")
    candidates = np.unique(np.r_[np.linspace(0, 1, 101), np.nextafter(1., 2.)])
    candidates_stats = []
    for threshold in candidates:
        stats, _, _ = event_metrics(predictions, events, threshold, cfg)
        if stats["events"] == 0:
            raise ValueError("Validation split needs evaluable awakenings")
        candidates_stats.append({"threshold": float(threshold), **stats})
    eligible = [s for s in candidates_stats if s["false_alarms_per_hour"] <= max_false_per_hour]
    # Prefer sensitivity, then fewer false alarms and less warning, then higher threshold.
    best = max(eligible, key=lambda s: (s["sensitivity"], -s["false_alarms_per_hour"],
                                       -s["time_in_warning_fraction"], s["threshold"]))
    return best["threshold"], candidates_stats


def subject_bootstrap(records, seed=42, repetitions=1000):
    grouped = records.groupby("subject")[["events", "detected", "false_alarms", "hours"]].sum()
    if len(grouped) < 2:
        return {"subjects": len(grouped), "sensitivity_95ci": None, "false_alarms_per_hour_95ci": None}
    rng = np.random.default_rng(seed)
    sensitivities, rates = [], []
    values = grouped.to_numpy()
    for _ in range(repetitions):
        e, d, f, h = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        if e:
            sensitivities.append(d/e)
        rates.append(f/h)
    return {"subjects": len(grouped),
            "sensitivity_95ci": np.quantile(sensitivities, [.025, .975]).tolist() if sensitivities else None,
            "false_alarms_per_hour_95ci": np.quantile(rates, [.025, .975]).tolist()}


def reliability_table(predictions, bins=10):
    frame = predictions[["label", "probability"]].copy()
    frame["bin"] = np.minimum((frame.probability*bins).astype(int), bins-1)
    return frame.groupby("bin").agg(count=("label", "size"), observed_rate=("label", "mean"),
                                    mean_probability=("probability", "mean")).reset_index()


def paired_subject_difference(baseline, joint, seed=42, repetitions=1000):
    """Paired participant bootstrap of sensitivity difference at separately locked thresholds."""
    a = baseline.groupby("subject")[["events", "detected"]].sum()
    b = joint.groupby("subject")[["events", "detected"]].sum().reindex(a.index)
    if not np.array_equal(a.events, b.events):
        raise ValueError("Paired comparison requires identical evaluable events")
    total = a.events.sum()
    point = float((b.detected.sum()-a.detected.sum())/total) if total else None
    if len(a) < 2 or not total:
        return {"joint_minus_spectral_sensitivity": point, "paired_subject_95ci": None}
    rng = np.random.default_rng(seed)
    values = np.column_stack([a.events, b.detected-a.detected])
    differences = []
    for _ in range(repetitions):
        events, delta = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        if events:
            differences.append(delta/events)
    return {"joint_minus_spectral_sensitivity": point,
            "paired_subject_95ci": np.quantile(differences, [.025, .975]).tolist(),
            "note": "Thresholds locked on validation; compare test false-alarm rates alongside sensitivity"}
