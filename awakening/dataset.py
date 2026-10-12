import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import feature_rows
from .io import load_recording, load_stages
from .labels import awakening_events, covered, future_label


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    records = spec["recordings"]
    if not spec.get("channels") or not records:
        raise ValueError("Manifest needs channels and recordings")
    ids, subjects, paths = set(), {}, set()
    for r in records:
        for key in ("id", "subject", "split", "signal", "annotations"):
            if not isinstance(r.get(key), str) or not r[key]:
                raise ValueError(f"Each recording requires string {key}")
        if r["id"] in ids or r["split"] not in {"train", "validation", "test"}:
            raise ValueError("Recording IDs must be unique; splits: train/validation/test")
        ids.add(r["id"])
        if r["subject"] in subjects and subjects[r["subject"]] != r["split"]:
            raise ValueError("Subject leakage: all nights of a subject must share a split")
        subjects[r["subject"]] = r["split"]
        for key in ("signal", "annotations"):
            r[key] = str((path.parent / r[key]).resolve())
            if not Path(r[key]).is_file():
                raise ValueError(f"File not found: {r[key]}")
        if r["signal"] in paths:
            raise ValueError("A signal cannot appear twice")
        paths.add(r["signal"])
    return spec


def build_dataset(manifest, cfg):
    spec = read_manifest(manifest)
    rows, event_rows, provenance = [], [], []
    digests = set()
    for r in spec["recordings"]:
        digest = sha256(r["signal"])
        if digest in digests:
            raise ValueError("Duplicate signal contents across recordings")
        digests.add(digest)
        rec = load_recording(r["signal"], spec["channels"], r.get("fs"), r.get("unit"), r.get("time_column"))
        stages = load_stages(r["annotations"], r.get("annotation_offset_s", 0))
        duration = len(rec.data) / rec.fs
        if (stages.onset + stages.duration).max() > duration + 1 / rec.fs:
            raise ValueError(f"Annotations extend beyond signal in {r['id']}; check clocks/offset")
        events = awakening_events(stages, cfg.min_sleep_s, cfg.min_wake_s)
        count = 0
        for t, features in feature_rows(rec, cfg):
            # Require full future observation to certify sustained wake and exclude right censoring.
            stop = t + cfg.lead_s + cfg.horizon_s + cfg.min_wake_s
            if stop > duration or not covered(stages, t-cfg.window_s, stop):
                continue
            rows.append({"recording": r["id"], "subject": r["subject"], "split": r["split"],
                         "time_s": t, "label": future_label(t, events, cfg), **features})
            count += 1
        if count == 0:
            raise ValueError(f"No evaluable windows in {r['id']}; check length and annotations")
        for onset in events:
            event_rows.append({"recording": r["id"], "subject": r["subject"],
                               "split": r["split"], "onset_s": onset})
        provenance.append({"id": r["id"], "subject": r["subject"], "split": r["split"],
                           "signal_sha256": digest, "annotation_sha256": sha256(r["annotations"]),
                           "fs": rec.fs, "duration_s": duration, "windows": count})
    event_frame = pd.DataFrame(event_rows, columns=["recording", "subject", "split", "onset_s"])
    return pd.DataFrame(rows), event_frame, {"channels": spec["channels"], "recordings": provenance,
                                           "reference": spec.get("reference", "not specified; verify montage"),
                                           "population": spec.get("population", "not specified"),
                                           "synthetic": bool(spec.get("synthetic", False))}
