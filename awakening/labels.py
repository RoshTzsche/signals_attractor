import numpy as np

SLEEP = {"N1", "N2", "N3", "REM"}
KNOWN = SLEEP | {"W"}


def state_runs(stages):
    """Merge contiguous sleep stages into sleep bouts, preserving gaps and unknowns."""
    runs = []
    for row in stages.itertuples():
        kind = "S" if row.stage in SLEEP else ("W" if row.stage == "W" else "?")
        start, end = float(row.onset), float(row.onset + row.duration)
        if runs and runs[-1][2] == kind and abs(runs[-1][1] - start) < 1e-6:
            runs[-1] = (runs[-1][0], end, kind)
        else:
            runs.append((start, end, kind))
    return runs


def awakening_events(stages, min_sleep_s, min_wake_s):
    runs = state_runs(stages)
    events = []
    for before, wake in zip(runs, runs[1:]):
        if (before[2] == "S" and wake[2] == "W" and abs(before[1] - wake[0]) < 1e-6
                and before[1] - before[0] >= min_sleep_s
                and wake[1] - wake[0] >= min_wake_s):
            events.append(wake[0])
    return np.asarray(events, dtype=float)


def covered(stages, start, end):
    """All of [start,end) must have known annotations, including the future target."""
    cursor = start
    for r in stages.itertuples():
        stop = r.onset + r.duration
        if stop <= cursor:
            continue
        if r.onset > cursor + 1e-6 or r.stage not in KNOWN:
            return False
        cursor = max(cursor, stop)
        if cursor >= end - 1e-6:
            return True
    return False


def future_label(t, events, cfg):
    return int(np.any((events >= t + cfg.lead_s) &
                      (events < t + cfg.lead_s + cfg.horizon_s)))
