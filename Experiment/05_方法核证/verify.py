"""Independent consistency checks for the immutable four-arm result."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "full_v2"


def main():
    manifest = json.loads((RUN / "freeze_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        digest = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        assert digest == item["sha256"], item["path"]
    predictions = pd.read_csv(RUN / "predictions.csv", dtype={"video_id": str})
    cases = pd.read_csv(RUN / "cases.csv", dtype={"video_id": str})
    matched = pd.read_csv(RUN / "event_matches.csv", dtype={"video_id": str})
    events = pd.read_csv(RUN / "predicted_events.csv", dtype={"video_id": str})
    metrics = pd.read_csv(RUN / "metrics.csv")
    assert len(predictions) == 4 * (49 + 73) + 49
    assert len(cases) == 47 * 4
    assert len(metrics) == 8
    for row in predictions.itertuples():
        assert row.predicted_count == len(str(row.peak_frames).split(";")) if row.predicted_count else pd.isna(row.peak_frames)
        assert np.isclose(row.predicted_rr_bpm, 60 * row.predicted_count / row.duration_seconds)
        if row.cohort != "source_sensitivity":
            actual = events[(events.video_id == row.video_id) & (events.cohort == row.cohort) & (events.arm == row.arm)]
            assert len(actual) == row.predicted_count
    for case in cases.itertuples():
        details = matched[(matched.video_id == case.video_id) & (matched.arm == case.arm)]
        tp, fp, fn = [sum(details.type == t) for t in ("TP", "FP", "FN")]
        assert (tp, fp, fn) == (case.tp, case.fp, case.fn)
        assert tp + fp == case.predicted_count
        assert tp + fn == case.truth_count
        assert details.loc[details.type.eq("TP"), "time_error_seconds"].le(.30 + 1e-9).all()
    for row in metrics.itertuples():
        group = predictions[(predictions.cohort == row.cohort) & (predictions.arm == row.arm) & predictions.truth_count.notna()]
        assert len(group) == row.n
        assert np.isclose(np.mean(abs(group.predicted_rr_bpm - group.truth_rr_bpm)), row.rr_mae_bpm)
        assert int(sum(group.predicted_count == group.truth_count)) == row.exact_count
    assert len(pd.read_csv(RUN / "B73_truth_discrepancies.csv")) == 4
    print("PASS: hashes, 537 predictions, 188 event-scored windows, counts, timing, metrics, B73 truth audit")


if __name__ == "__main__":
    main()
